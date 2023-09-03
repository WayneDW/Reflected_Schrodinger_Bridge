import ml_collections

def get_gmm_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.seed=42 #The utlimate answer of UNIVERSE!
  config.T = 1.0
  config.interval = 100
  config.train_method = 'alternate'
  #config.t0 = 1e-3 ####### simple sde is 0, now set to 1e-3 following MNIST config file # this part is quite quite important to avoid nan loss in DSM-warm_up
  config.t0 = 1e-5
  config.problem_name = 'gmm'
  config.num_itr = 500
  config.eval_itr = 200
  config.forward_net = 'toy'
  config.backward_net = 'toy'
  config.use_arange_t = True
  config.num_epoch = 1
  config.num_stage = 8
  config.train_bs_x = 1024
  config.sde_type = 'vp'
  # sampling
  config.samp_bs = 4000
  config.sigma_min = 0.1
  config.sigma_max = 2
  config.snapshot_freq = 1
  # optimization
  # config.optim = optim = ml_collections.ConfigDict()
  config.weight_decay = 0
  config.optimizer = 'AdamW'
  config.lr = 1e-4
  config.lr_gamma = 0.9
  # domain setting
  config.domain_name = 'Polygon'
  config.domain_radius = 13.

  model_configs=None


  """ alternative hyperparameters """
  config.train_bs_x_dsm = 192
  config.train_bs_t_dsm = 192
  #config.train_bs_t = 4
  config.num_itr_dsm = 10000 # # 5000 #100 # 5000
  config.DSM_warmup = True
  return config, model_configs
