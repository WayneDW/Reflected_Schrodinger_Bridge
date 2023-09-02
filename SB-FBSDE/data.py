import numpy as np

import torch
import torch.distributions as td
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from prefetch_generator import BackgroundGenerator

import util
from ipdb import set_trace as debug

from domain import Flower, Polygon, Heart, Cross, Star
from tools import HelperTorch, Sampler

def build_boundary_distribution(opt):
    print(util.magenta("build boundary distribution..."))

    opt.data_dim = get_data_dim(opt.problem_name)
    prior = build_prior_sampler(opt, opt.samp_bs)
    pdata = build_data_sampler(opt, opt.samp_bs)

    return pdata, prior

def get_data_dim(problem_name):
    return {
        'gmm':          [2],
        'checkerboard': [2],
        'moon-to-spiral':[2],
    }.get(problem_name)

def build_prior_sampler(opt, batch_size):
    if opt.problem_name == 'moon-to-spiral':
        # 'moon-to-spiral' uses Moon as prior distribution
        return Moon(batch_size)

    prior = td.MultivariateNormal(torch.zeros(opt.data_dim), torch.eye(opt.data_dim[-1]))
    return PriorSampler(prior, batch_size, opt)

def build_data_sampler(opt, batch_size):
    if util.is_toy_dataset(opt):
        return {
            'gmm': MixMultiVariateNormal,
            'checkerboard': CheckerBoard,
            'moon-to-spiral': Spiral,
        }.get(opt.problem_name)(batch_size, opt)
    else:
        raise RuntimeError()

def get_domain(opt):

    DomainClasses = {'Flower': Flower,
                     'Polygon': Polygon,
                     'Heart': Heart,
                     'Cross': Cross,
                     'Star': Star}
    if opt.domain_name in DomainClasses:
        return DomainClasses.get(opt.domain_name)
    else:
        raise RuntimeError()

def filter_outside_domain(myHelper, sample, device):
    constrain_sample = torch.empty((0, 2)).to(device)
    for idx in range(sample.shape[0]):
        inside_domain = myHelper.inside_domain(sample[idx, :])
        if inside_domain:
            constrain_sample = torch.cat((constrain_sample, sample[idx, :].reshape(1, -1)), dim=0)
    return constrain_sample


class MixMultiVariateNormal:
    def __init__(self, batch_size, opt, radius=12, num=8, sigmas=None):

        # build mu's and sigma's
        arc = 2 * np.pi / num
        xs = [np.cos(arc*idx)*radius for idx in range(num)]
        ys = [np.sin(arc*idx)*radius for idx in range(num)]
        mus = [torch.Tensor([x,y]) for x,y in zip(xs,ys)]
        dim = len(mus[0])
        sigmas = [torch.eye(dim) for _ in range(num)] if sigmas is None else sigmas

        if batch_size%num!=0:
            raise ValueError('batch size must be devided by number of gaussian')
        self.num = num
        self.batch_size = batch_size
        self.dists=[
            td.multivariate_normal.MultivariateNormal(mu, sigma) for mu, sigma in zip(mus, sigmas)
        ]

        self.device = opt.device
        self.myHelper = HelperTorch(get_domain(opt), self.device, max_radius=opt.domain_radius)

    def log_prob(self,x):
        # assume equally-weighted
        densities = [torch.exp(dist.log_prob(x)) for dist in self.dists]
        return torch.log(sum(densities)/len(self.dists))

    def sample(self):
        ind_sample = self.batch_size * 2 / self.num
        samples = [dist.sample([int(ind_sample)]) for dist in self.dists]
        sample = torch.cat(samples,dim=0)

        sample = torch.Tensor(sample).to(self.device)
        sample = filter_outside_domain(self.myHelper, sample, self.device)
        return sample[0:self.batch_size,:]

class CheckerBoard: # constrained version
    def __init__(self, batch_size, opt):
        self.batch_size = batch_size
        self.device = opt.device
        self.myHelper = HelperTorch(get_domain(opt), self.device, max_radius=opt.domain_radius)
        

    def sample(self):
        n = self.batch_size * 2 # make it redundante to satify the filter in the final step
        n_points = 3 * n
        n_classes = 2
        freq = 5
        x = np.random.uniform(-(freq//2)*np.pi, (freq//2)*np.pi, size=(n_points, n_classes))
        mask = np.logical_or(np.logical_and(np.sin(x[:,0]) > 0.0, np.sin(x[:,1]) > 0.0), \
        np.logical_and(np.sin(x[:,0]) < 0.0, np.sin(x[:,1]) < 0.0))
        y = np.eye(n_classes)[1*mask]
        x0 = x[:,0] * y[:,0]
        x1 = x[:,1] * y[:,0]
        sample = np.concatenate([x0[...,None],x1[...,None]],axis=-1)
        sqr = np.sum(np.square(sample),axis=-1)
        idxs = np.where(sqr==0)
        sample = np.delete(sample,idxs,axis=0)
        sample = torch.Tensor(sample).to(self.device)
        sample = filter_outside_domain(self.myHelper, sample, self.device)
        return sample[0:self.batch_size,:]

class Spiral:
    def __init__(self, batch_size, opt):
        self.batch_size = batch_size
        self.device = opt.device
        self.myHelper = HelperTorch(get_domain(opt), self.device, max_radius=opt.domain_radius)

    def sample(self):
        n = self.batch_size * 2 # to increase the total samples
        theta = np.sqrt(np.random.rand(n))*3*np.pi-0.5*np.pi # np.linspace(0,2*pi,100)

        r_a = theta + np.pi
        data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
        x_a = data_a + 0.25*np.random.randn(n,2)
        samples = np.append(x_a, np.zeros((n,1)), axis=1)
        samples = samples[:,0:2]
        sample = torch.Tensor(sample).to(self.device)
        sample = filter_outside_domain(self.myHelper, sample, self.device)
        return sample[0:self.batch_size,:]

class Moon:
    def __init__(self, batch_size, opt):
        self.batch_size = batch_size
        self.device = opt.device
        self.myHelper = HelperTorch(get_domain(opt), self.device, max_radius=opt.domain_radius)

    def sample(self):
        n = self.batch_size #* 5 # to increase the total samples
        x = np.linspace(0, np.pi, n // 2)
        u = np.stack([np.cos(x) + .5, -np.sin(x) + .2], axis=1) * 10.
        u += 0.5*np.random.normal(size=u.shape)
        v = np.stack([np.cos(x) - .5, np.sin(x) - .2], axis=1) * 10.
        v += 0.5*np.random.normal(size=v.shape)
        x = np.concatenate([u, v], axis=0)
        sample = torch.Tensor(x).to(self.device)
        sample = filter_outside_domain(self.myHelper, sample, self.device)
        return sample[0:self.batch_size,:]

class PriorSampler: # a dump prior sampler to align with DataSampler
    def __init__(self, prior, batch_size, opt):
        self.prior = prior
        self.batch_size = batch_size
        self.device = opt.device
        self.myHelper = HelperTorch(get_domain(opt), self.device, max_radius=opt.domain_radius)

    def log_prob(self, x):
        return self.prior.log_prob(x)

    def sample(self):
        sample = self.prior.sample([self.batch_size*2])
        sample = filter_outside_domain(self.myHelper, sample, self.device)
        return sample[0:self.batch_size, :]
