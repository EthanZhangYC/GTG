import os
import torch

from params_proto.neo_proto import ParamsProto, PrefixProto, Proto

class Config(ParamsProto):
    # misc
    seed = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bucket = 'trained_models/'
    dataset = 'ant'

    ## model
    model = 'models.TemporalUnet'
    diffusion = 'models.GaussianDiffusion'
    horizon = 64
    n_diffusion_steps = 200
    n_sample_timesteps = 200
    action_weight = 10
    loss_weights = None
    loss_discount = 1
    predict_epsilon = True
    dim_mults = (1, 4, 8)
    returns_condition = False
    calc_energy = False
    dim = 128
    condition_dropout = 0.25
    condition_guidance_w = 1.2
    test_ret = 0.9
    renderer = None

    ## dataset
    loader = 'datasets.PointRegretDataset'
    proxy_loader = 'datasets.ZipDataset'
    data_path = 'generated_datasets/'
    context_length = 32
    regret = False
    include_returns = False
    
    # normalizer = 'CDFNormalizer'
    # preprocess_fns = []
    clip_denoised = True
    # use_padding = True
    # include_returns = True
    # discount = 0.99
    # max_path_length = 1000
    # hidden_dim = 256
    # ar_inv = False
    train_only_inv = False
    # termination_penalty = -100
    # returns_scale = 400.0 # Determined using rewards from the dataset

    ## training
    n_steps_per_epoch = 10
    batch_size = 1024
    loss_type = 'l2'
    n_train_steps = 300000
    learning_rate = 1e-4
    gradient_accumulate_every = 2
    ema_decay = 0.995
    log_freq = 50
    save_freq = 1000
    sample_freq = 10000
    n_saves = 5
    save_parallel = False
    n_reference = 8
    save_checkpoints = True
    
    ## proxy_model
    proxy_model = "models.Proxy"
    proxy_hidden_dim = 1024
    proxy_n_ensembles = 10
    proxy_learning_rate = 1e-3
    proxy_n_train_steps = 5000
    proxy_log_freq = 100
    proxy_save_freq = 1000
