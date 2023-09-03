import ml_collections

def get_checkerboard_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.seed = 42 #The utlimate answer of UNIVERSE!
  config.T = 1.0
  config.interval = 100
  config.train_method = 'alternate'
  config.t0 = 3e-5
  config.problem_name = 'checkerboard'
  config.num_itr = 800
  config.eval_itr = 200
  config.forward_net = 'toy'
  config.backward_net = 'toy'
  config.use_arange_t = True
  config.num_epoch = 1
  config.num_stage = 20
  config.train_bs_x = 1000
  config.sde_type = 'vp'
  # sampling
  config.samp_bs = 4000
  config.sigma_min = 0.2
  config.sigma_max = 1
  config.beta_min = 0.04
  config.beta_max = 0.5
  config.snapshot_freq = 2
  # optimization
  # config.optim = optim = ml_collections.ConfigDict()
  config.weight_decay = 0
  config.optimizer = 'AdamW'
  config.lr = 6e-4
  config.lr_gamma = 0.8

  # domain settings
  config.domain_name = 'Flower'
  config.domain_radius = 7.

  model_configs=None

  return config, model_configs
