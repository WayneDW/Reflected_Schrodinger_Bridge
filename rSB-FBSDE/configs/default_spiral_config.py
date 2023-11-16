import ml_collections

def get_spiral_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.seed=42 #The utlimate answer of UNIVERSE!
  config.T = 1.0
  config.interval = 100
  config.train_method = 'alternate'
  config.t0 = 0
  config.problem_name = 'spiral'
  config.num_itr = 2000
  config.eval_itr = 200
  config.forward_net = 'toy'
  config.backward_net = 'toy'
  config.use_arange_t = True
  config.num_epoch = 1
  config.num_stage = 5
  config.train_bs_x = 1000 
  config.sde_type = 'vp' # 'simple' # worked # 'vp' # cause vanilla SB with Linear training to fail
  # sampling
  config.samp_bs = 1000
  config.sigma_min = 0.1
  config.sigma_max = 2
  config.beta_min = 0.1 # 0.03 # the original 0.1/ 20 could recover the shape
  config.beta_max = 10 # 2
  config.snapshot_freq = 1
  # optimization
#   config.optim = optim = ml_collections.ConfigDict()
  config.weight_decay = 0
  config.optimizer = 'AdamW'
  config.lr = 3e-4
  config.lr_gamma = 0.9

  model_configs=None

  # domain settings
  config.domain_name = 'Flower'
  config.domain_radius = 12.5

  """ alternative hyperparameters """
  config.train_bs_x_dsm = 200
  config.train_bs_t_dsm = 200
  #config.train_bs_t = 4
  config.num_itr_dsm = 1000 # # 5000 #100 # 5000
  config.DSM_warmup = False #True

  return config, model_configs
