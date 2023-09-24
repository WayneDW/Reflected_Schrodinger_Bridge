import os,sys,re

import numpy as np
import shutil
import termcolor
import pathlib
from scipy import linalg
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.utils as tu
from torch.nn.functional import adaptive_avg_pool2d

import pandas as pd
import seaborn
from scipy import stats

from data import get_domain
from domain import Flower, Polygon, Heart, Cross, Star
from tools import HelperTorch

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from ipdb import set_trace as debug


# convert to colored strings
def red(content): return termcolor.colored(str(content),"red",attrs=["bold"])
def green(content): return termcolor.colored(str(content),"green",attrs=["bold"])
def blue(content): return termcolor.colored(str(content),"blue",attrs=["bold"])
def cyan(content): return termcolor.colored(str(content),"cyan",attrs=["bold"])
def yellow(content): return termcolor.colored(str(content),"yellow",attrs=["bold"])
def magenta(content): return termcolor.colored(str(content),"magenta",attrs=["bold"])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def is_toy_dataset(opt):
    return opt.problem_name in ['gmm','checkerboard', 'moon-to-spiral', 'smile-to-checkerboard']

def use_vp_sde(opt):
    return opt.sde_type == 'vp'

def evaluate_stage(opt, stage, metrics):
    """ Determine what metrics to evaluate for the current stage,
    if metrics is None, use the frequency in opt to decide it.
    """
    if metrics is not None:
        return [k in metrics for k in ['FID', 'snapshot', 'ckpt']]
    match = lambda freq: (freq>0 and stage%freq==0)
    return [match(opt.FID_freq), match(opt.snapshot_freq), match(opt.ckpt_freq)]

def get_time(sec):
    h = int(sec//3600)
    m = int((sec//60)%60)
    s = sec%60
    return h,m,s

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def flatten_dim01(x):
    # (dim0, dim1, *dim2) --> (dim0x1, *dim2)
    return x.reshape(-1, *x.shape[2:])

def unflatten_dim01(x, dim01):
    # (dim0x1, *dim2) --> (dim0, dim1, *dim2)
    return x.reshape(*dim01, *x.shape[1:])

def compute_z_norm(zs, dt):
    # Given zs.shape = [batch, timesteps, *z_dim], return E[\int 0.5*norm(z)*dt],
    # where the norm is taken over z_dim, the integral is taken over timesteps,
    # and the expectation is taken over batch.
    zs = zs.reshape(*zs.shape[:2],-1)
    return 0.5 * zs.norm(dim=2).sum(dim=1).mean(dim=0) * dt

def create_traj_sampler(trajs):
    for traj in trajs:
        yield traj

def get_load_it(load_name):
    nums = re.findall('[0-9]+', load_name)
    assert len(nums)>0
    if 'stage' in load_name and 'dsm' in load_name:
        return int(nums[-2])
    return int(nums[-1])

def restore_checkpoint(opt, runner, load_name):
    assert load_name is not None
    print(green("#loading checkpoint {}...".format(load_name)))

    if 'checkpoint_16.pth' in load_name:
        # loading pre-trained NCSN++ from
        # https://drive.google.com/drive/folders/1sP4GwvrYiI-sDPTp7sKYzsxJLGVamVMZ
        assert opt.backward_net == 'ncsnpp'

        with torch.cuda.device(opt.gpu):
            checkpoint = torch.load(load_name)
            model_ckpt, ema_params_ckpt = checkpoint['model'], checkpoint['ema']['shadow_params']

            # load model
            res = {k.replace('module.', 'net.') : v for k, v in model_ckpt.items()}
            runner.z_b.load_state_dict(res) # Dont load key:sigmas.
            print(green('#successfully loaded all the modules'))

            # load ema
            assert type(runner.ema_b.shadow_params) == list
            runner.ema_b.shadow_params = [p.to(opt.device) for p in ema_params_ckpt]
            print(green('#loading form ema shadow parameter for polices'))

    else:
        full_keys = ['z_f','optimizer_f','ema_f','z_b','optimizer_b','ema_b']

        with torch.cuda.device(opt.gpu):
            checkpoint = torch.load(load_name,map_location=opt.device)
            ckpt_keys=[*checkpoint.keys()]
            for k in ckpt_keys:
                getattr(runner,k).load_state_dict(checkpoint[k])

        if len(full_keys)!=len(ckpt_keys):
            value = { k for k in set(full_keys) - set(ckpt_keys) }
            print(green("#warning: does not load model for {}, check is it correct".format(value)))
        else:
            print(green('#successfully loaded all the modules'))

        # Note: Copy the avergage parameter to policy. This seems to improve performance for
        # DSM warmup training (yet not sure whether it's true for SB in general)
        runner.ema_f.copy_to()
        runner.ema_b.copy_to()
        print(green('#loading form ema shadow parameter for polices'))
    print(magenta("#######summary of checkpoint##########"))

def save_checkpoint(opt, runner, keys, stage_it, dsm_train_it=None):
    checkpoint = {}
    #fn = opt.ckpt_path + "/stage_{0}{1}.npz".format(
    #    stage_it, '_dsm{}'.format(dsm_train_it) if dsm_train_it is not None else ''
    #)
    fn = opt.ckpt_path + f"/{opt.problem_name}_stage_{stage_it}_seed_{opt.seed}.npz"
    with torch.cuda.device(opt.gpu):
        for k in keys:
            checkpoint[k] = getattr(runner,k).state_dict()
        torch.save(checkpoint, fn)
    print(green("checkpoint saved: {}".format(fn)))

def save_toy_npy_traj(opt, fn, traj, n_snapshot=None, direction=None):
    #form of traj: [bs, interval, x_dim=2]
    #fn_npy = os.path.join('results', opt.dir, fn+'.npy')
    fn_pdf = os.path.join('results', opt.dir, f'{opt.problem_name}_{fn}.png')

    xlims = {
        'gmm': [-17, 17],
        'checkerboard': [-8.5, 8.5],
        'moon-to-spiral':[-12, 12],
        'smile-to-checkerboard': [-8.5, 8.5],
    }.get(opt.problem_name)

    ylims = {
        'gmm': [-17, 17],
        'checkerboard': [-8, 9],
        'moon-to-spiral':[-13, 9],
        'smile-to-checkerboard': [-8.5, 8.5],
    }.get(opt.problem_name)

    myDomainCurve = get_domain(opt)(radius=opt.domain_radius).position(np.arange(0, 1, 0.001))

    if n_snapshot is None: # only store t=0
        plt.scatter(traj[:,0,0],traj[:,0,1], s=5)
        plt.xlim(*xlims)
        plt.ylim(*ylims)
    else:
        total_steps = traj.shape[1]
        sample_steps = np.linspace(0, total_steps-1, n_snapshot).astype(int)
        color = 'salmon' if direction=='forward' else 'royalblue'
        ''' # original plot
        fig, axs = plt.subplots(1, n_snapshot)
        fig.set_size_inches(n_snapshot*6, 6)
        for ax, step in zip(axs, sample_steps):
            ax.scatter(traj[:,step,0],traj[:,step,1], s=5, color=color)
            ax.scatter(myDomainCurve[0, :],myDomainCurve[1, :], s=5, alpha=0.05, color=color)
            ax.set_xlim(*xlims)
            ax.set_ylim(*ylims)
            ax.set_title('time = {:.2f}'.format(step/(total_steps-1)*opt.T))
        fig.tight_layout()
        '''
        """ Test Yu's new plot """
        num_row = 1
        num_col = np.ceil(n_snapshot/num_row).astype(int)
        plt.style.use('default')
        fig, axes = plt.subplots(num_row, num_col, figsize=[num_col*2.5, num_row*2.5])
        show_boundary = True
        if show_boundary:
            plt.subplots_adjust(hspace=0.0, wspace=0.01)
        else:
            plt.subplots_adjust(hspace=0.0, wspace=0.0) # Yu's original
        #fig.patch.set_facecolor('lightgrey') # Yu's original
        axes = axes.reshape(-1)

        for ax, step in zip(axes, sample_steps):
            x, y = traj[:, step, 0], traj[:, step, 1]
            values = np.vstack([x, y])
            kernel = stats.gaussian_kde(values)(values)

            ax = fig.add_subplot(ax)
            cmap = 'flare' if direction == 'forward' else 'crest'
            bcol = 'salmon' if direction == 'forward' else 'darkgreen'
            seaborn.scatterplot(x=x, y=y, s=4, c=kernel, vmin=-0.002, cmap=cmap, alpha=1.0)
            seaborn.scatterplot(x=myDomainCurve[0, :], y=myDomainCurve[1, :], s=5, color=bcol, alpha=1.0)
            ax.set_xlim(*xlims)
            ax.set_ylim(*ylims)
            if show_boundary:
                plt.tick_params(left = False, right = False , labelleft = False ,
                        labelbottom = False, bottom = False)
                plt.setp(ax.spines.values(), color='grey')
            else:  
                ax.axis('off')
        #plt.axis('off')
        fig.tight_layout()
    plt.savefig(fn_pdf, dpi=250)
    #np.save(fn_npy, traj)
    plt.clf()

def get_FID_npz_path(opt):
    if opt.FID_ckpt is not None: return opt.FID_ckpt
    return {
        'cifar10': 'checkpoint/cifar10_fid_stat_local.npz',
    }.get(opt.problem_name, None)

def snapshot(opt, img, stage, direction):

    t=-1 if direction=='forward' else 0
    n = 36 if opt.compute_FID else 24

    img = img[0:n,t,...] if len(img.shape)==5 else img[0:n,...]
    img=norm_data(opt, img) #Norm data to [0,1]

    fn = os.path.join(
        opt.eval_path,
        direction,
        '{}stage{}.png'.format('sample_' if opt.compute_FID else '', stage)
    )
    tu.save_image(img, fn, nrow = 6)

def save_generated_data(opt, x):
    x = norm_data(opt,x)
    x = torch.squeeze(x)
    for i in range(x.shape[0]):
        fn = os.path.join(opt.generated_data_path, 'img{}.jpg'.format(i))
        tu.save_image(x[i,...], fn)

def compute_fid(opt, xTs):
    FID_path = get_FID_npz_path(opt)
    save_generated_data(opt, xTs.to(opt.device))
    return get_fid(FID_path, opt.generated_data_path)

def exist_FID_ckpt(opt):
    ckpt = get_FID_npz_path(opt)
    return ckpt is not None and os.path.exists(ckpt)

def norm_data(opt,x):
    if opt.problem_name=='mnist':
        x=x.repeat(1,3,1,1)
    _max=torch.max(torch.max(x,dim=-1)[0],dim=-1)[0][...,None,None]
    _min=torch.min(torch.min(x,dim=-1)[0],dim=-1)[0][...,None,None]
    x=(x-_min)/(_max-_min)
    return x

def check_duplication(opt):
    plt_dir='plots/'+opt.dir
    ckpt_dir='checkpoint/'+opt.group+'/'+opt.name
    runs_dir='runs/'+opt.log_fn
    plt_flag=os.path.isdir(plt_dir)
    ckpt_flag=os.path.isdir(ckpt_dir)
    run_flag=os.path.isdir(runs_dir)
    tot_flag= plt_flag or ckpt_flag or run_flag
    print([plt_flag,ckpt_flag,run_flag])
    if tot_flag:
        decision=input('Exist duplicated folder, do you want to overwrite it? [y/n]')

        if 'y' in decision:
            try:
                shutil.rmtree(plt_dir)
            except:
                pass
            try: 
                shutil.rmtree(ckpt_dir)
            except:
                pass
            try:
                shutil.rmtree(runs_dir)
            except:
                pass
        else:
            sys.exit()

######################################################################################
##                          Copy of FID computation utils                           ##
##  Ref: https://github.com/ermongroup/ncsnv2/blob/master/evaluation/fid_score.py,  ##
##       https://github.com/ermongroup/ncsnv2/blob/master/evaluation/inception.py,  ##
######################################################################################

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def _compute_statistics_of_path(path, model, batch_size, dims, cuda):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, cuda)

    return m, s

def calculate_fid_npz(path, root, name, batch_size=256, cuda=True, dims=2048):
    """Calculates the FID of two paths"""

    from models.InceptionNet.inception_net import InceptionV3

    if not os.path.exists(path):
        raise RuntimeError('Invalid path: %s' % path)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()
    m1, s1 = _compute_statistics_of_path(path, model, batch_size,
                                         dims, cuda)

    if not os.path.exists(root):
        os.makedirs(root)
    np.savez(root+name, mu=m1, sigma=s1)

def calculate_fid_given_paths(paths, batch_size, cuda, dims):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    from models.InceptionNet.inception_net import InceptionV3

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()
    m1, s1 = _compute_statistics_of_path(paths[0], model, batch_size,
                                         dims, cuda)

    m2, s2 = _compute_statistics_of_path(paths[1], model, batch_size,
                                         dims, cuda)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def get_fid(path1, path2):
    fid_value = calculate_fid_given_paths([path1, path2],
                                          256,
                                          True,
                                          2048)
    return fid_value

def get_fid_stats_path(args, config, download=True):

    links = {
        'CIFAR10': 'http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_cifar10_train.npz',
        'LSUN': 'http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_lsun_train.npz'
    }
    if config.data.dataset == 'CIFAR10':
        path = os.path.join(args.exp, 'datasets', 'cifar10_fid.npz')
        if not os.path.exists(path):
            if not download:
                raise FileNotFoundError("no statistics file founded")
            else:
                import urllib
                urllib.request.urlretrieve(
                    links[config.data.dataset], path
                )
    elif config.data.dataset == 'CELEBA':
        path = os.path.join(args.exp, 'datasets', 'celeba_test_fid_stats.npz')
        if not os.path.exists(path):
            raise FileNotFoundError('no statistics file founded')

    return path
