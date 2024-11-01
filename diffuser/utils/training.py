import os
import copy
import numpy as np
import torch
import einops
import pdb
import diffuser
from copy import deepcopy
# import wandb
from tqdm import tqdm

from .arrays import batch_to_device, to_np, to_device, apply_dict, to_torch
from .timer import Timer
from .cloud import sync_logs
from ml_logger import logger

import sys
import os.path as osp
import time
import numpy as np
import pickle
import pdb

import torch
import torch.nn as nn
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
    

        


def get_label(single_dataset,idx,label_dict):
    label = single_dataset[idx][1].item()
    return label

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
        
        
        label_dict={'0':0,'1':0,'2':1,'3':1,'4':1}
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx, label_dict)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        
        weights = [1.0 / label_to_count[self._get_label(dataset, idx, label_dict)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx, label_dict):
        if self.callback_get_label:
            return self.callback_get_label(dataset, idx, label_dict)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        else:
            raise NotImplementedError

    # sample class balance training batch 
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)

def filter_area(trajs, labels, pad_masks):
    new_list=[]
    new_list_y=[]
    lat_min,lat_max = (18.249901, 55.975593)
    lon_min,lon_max = (-122.3315333, 126.998528)
    len_traj = trajs.shape[0]
    # avg_lat_list=[]
    # avg_lon_list=[]
    for i in range(len_traj):
        traj = trajs[i]
        pad_mask = pad_masks[i]
        label = labels[i]
        
        new_traj = traj[~pad_mask][:,:2]
        new_traj[:,0] = new_traj[:,0] * (lat_max-lat_min) + lat_min
        new_traj[:,1] = new_traj[:,1] * (lon_max-lon_min) + lon_min
        avg_lat, avg_lon = np.mean(new_traj, axis=0)
        
        # avg_lat_list.append(avg_lat)
        # avg_lon_list.append(avg_lon)
        if avg_lat<41 and avg_lat>39:
            if avg_lon>115 and avg_lon<117:
                new_list.append(traj)
                new_list_y.append(label)
                
    return np.array(new_list), np.array(new_list_y)

def generate_posid(trajs, pad_masks, min_max=[(18.249901, 55.975593),(-122.3315333, 126.998528)]):
    lat_min,lat_max = min_max[0]
    lon_min,lon_max = min_max[1]
    
    new_list=[]
    new_list_y=[]
    len_traj = trajs.shape[0]
    
    max_list_lat=[]
    max_list_lon=[]
    min_list_lat=[]
    min_list_lon=[]
    for i in range(len_traj):
        traj = trajs[i]
        pad_mask = pad_masks[i]
        new_traj = traj[~pad_mask][:,:2]
        new_traj[:,0] = new_traj[:,0] * (lat_max-lat_min) + lat_min
        new_traj[:,1] = new_traj[:,1] * (lon_max-lon_min) + lon_min
        tmp_max_lat,tmp_max_lon = np.max(new_traj, axis=0)
        tmp_min_lat,tmp_min_lon = np.min(new_traj, axis=0)
        max_list_lat.append(tmp_max_lat)
        max_list_lon.append(tmp_max_lon)
        min_list_lat.append(tmp_min_lat)
        min_list_lon.append(tmp_min_lon)
    
    tmp_max_lat = np.max(np.array(max_list_lat))+1e-6
    tmp_max_lon = np.max(np.array(max_list_lon))+1e-6
    tmp_min_lat = np.min(np.array(min_list_lat))-1e-6
    tmp_min_lon = np.min(np.array(min_list_lon))-1e-6
    print(tmp_max_lat,tmp_max_lon,tmp_min_lat,tmp_min_lon)
        
    # tmp_max_lat,tmp_max_lon,tmp_min_lat,tmp_min_lon = 40.8855, 117.2707, 38.44578, 114.93135
    patchlen_lat = (tmp_max_lat-tmp_min_lat) / 16
    patchlen_lon = (tmp_max_lon-tmp_min_lon) / 16
    sid_list=[]
    eid_list=[]
    for i in range(len_traj):
        traj = trajs[i]
        pad_mask = pad_masks[i]
        # label = labels[i]
        
        new_traj = traj[~pad_mask][:,:2]
        new_traj[:,0] = new_traj[:,0] * (lat_max-lat_min) + lat_min
        new_traj[:,1] = new_traj[:,1] * (lon_max-lon_min) + lon_min
        # avg_lat, avg_lon = np.mean(new_traj, axis=0)
        
        sid = (new_traj[0,0]-tmp_min_lat)//patchlen_lat*16+(new_traj[0,1]-tmp_min_lon)//patchlen_lon
        eid = (new_traj[-1,0]-tmp_min_lat)//patchlen_lat*16+(new_traj[-1,1]-tmp_min_lon)//patchlen_lon
        sid_list.append(sid)
        eid_list.append(eid)
        # if sid>=256 or eid>=256:
        #     pdb.set_trace()

    return np.array(sid_list), np.array(eid_list)

def load_data(batch_sizes, traj_length):
    # batch_sizes = config.training.batch_size
    filename = '/home/yichen/TS2Vec/datafiles/Geolife/traindata_4class_xy_traintest_interpolatedNAN_5s_trip20_new_001meters_withdist_aligninterpolation_InsertAfterSeg_Both_11dim_doubletest_0218.pickle'
    with open(filename, 'rb') as f:
        kfold_dataset, X_unlabeled = pickle.load(f)
    dataset = kfold_dataset

    train_x = dataset[1].squeeze(1)
    train_y = dataset[3]
    train_x = train_x[:,:,4:]   
    pad_mask_source = train_x[:,:,0]==0
    train_x[pad_mask_source] = 0.
    
    # if config.data.interpolated:
    train_x_ori = dataset[1].squeeze(1)[:,:,2:]
    # else:
    # train_x_ori = dataset[0].squeeze(1)[:,:,2:]
    train_y_ori = dataset[3]
    pad_mask_source_train_ori = train_x_ori[:,:,2]==0
    train_x_ori[pad_mask_source_train_ori] = 0.
    
    # class_id = 2
    # print('filtering class %d'%class_id)
    # mask_class = train_y_ori==class_id
    # train_x_ori = train_x_ori[mask_class] 
    # train_y_ori = train_y_ori[mask_class] 
    
    # if config.data.filter_area:
    print('filtering area')
    train_x_ori,train_y_ori = filter_area(train_x_ori, train_y_ori, pad_mask_source_train_ori)
    pad_mask_source_train_ori = train_x_ori[:,:,2]==0


    if traj_length < train_x_ori.shape[1]:
        train_x_ori = train_x_ori[:,:traj_length,:]
        pad_mask_source_train_ori = pad_mask_source_train_ori[:,:traj_length]

    # if "seid" in config.model.mode:
    sid,eid = generate_posid(train_x_ori, pad_mask_source_train_ori)
    se_id = np.stack([sid, eid]).T
    
    print('filtering nopadding segments')
    pad_mask_source_incomplete = np.sum(pad_mask_source_train_ori,axis=1) == 0
    train_x_ori = train_x_ori[pad_mask_source_incomplete]
    train_y_ori = train_y_ori[pad_mask_source_incomplete]
    se_id = se_id[pad_mask_source_incomplete]
        
    class_dict={}
    for y in train_y:
        if y not in class_dict:
            class_dict[y]=1
        else:
            class_dict[y]+=1
    print('Geolife:',dict(sorted(class_dict.items())))
    print('Reading Data: (train: geolife + MTL, test: MTL)')
    print('GeoLife shape: '+str(train_x_ori.shape))
    n_geolife = train_x.shape[0]
    train_dataset_geolife = TensorDataset(
        torch.from_numpy(train_x).to(torch.float),
        torch.from_numpy(train_y),
        torch.from_numpy(np.array([0]*n_geolife)).float()
    )
    
        
    train_dataset_ori = TensorDataset(
        torch.from_numpy(train_x_ori).to(torch.float),
        torch.from_numpy(se_id).to(torch.float),
        torch.from_numpy(train_y_ori),
        # torch.from_numpy(np.array([{"ctx_len": np.array([0])}]*n_geolife))
    )
    train_loader_source_ori = DataLoader(train_dataset_ori, batch_size=min(batch_sizes, len(train_dataset_geolife)), num_workers=0, shuffle=True, drop_last=False)
    train_tgt_iter = ForeverDataIterator(train_loader_source_ori)
    return train_tgt_iter







class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        proxy_model,
        dataset,
        proxy_dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        proxy_train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        proxy_log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        proxy_save_freq=100,
        label_freq=100000,
        save_parallel=False,
        n_reference=8,
        bucket=None,
        train_device='cuda',
        save_checkpoints=False,
        load_checkpoint=None,
        load_proxy_checkpoint=None,
        horizon=64
    ):
        super().__init__()
        self.model = diffusion_model
        self.proxy_model = proxy_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.save_checkpoints = save_checkpoints

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.proxy_log_freq = proxy_log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.proxy_save_freq = proxy_save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.proxy_dataset = proxy_dataset
        
        self.dataloader = load_data(train_batch_size, horizon)
        self.horizon = horizon
        
        # ranks = torch.argsort(torch.argsort(-1 * self.proxy_dataset.data_y.flatten()))
        # weights = 1.0 / (1e-2 * len(self.proxy_dataset.data_y) + ranks)
        # sampler = torch.utils.data.WeightedRandomSampler(
        #         weights=weights, num_samples=len(self.proxy_dataset.data_y), replacement=True
        #         )
        # self.dataloader = cycle(torch.utils.data.DataLoader(
        #     self.dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True, drop_last=True,
        # ))
        # self.proxy_dataloader = cycle(torch.utils.data.DataLoader(
        #     self.proxy_dataset, batch_size=train_batch_size, num_workers=0, sampler=sampler, pin_memory=True, drop_last=True,
        # ))
        # self.proxy_dataloader = cycle(torch.utils.data.DataLoader(
        #     self.proxy_dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True, drop_last=True,
        # ))
        
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)
        self.proxy_optimizer = torch.optim.Adam(proxy_model.parameters(), lr=proxy_train_lr)

        self.bucket = bucket
        self.n_reference = n_reference

        self.reset_parameters()
        self.step = 0
        self.proxy_step = 0

        self.device = train_device
        
        if load_checkpoint is not None:
            self.load(epoch=load_checkpoint)
            self.step = load_checkpoint
            
        if load_proxy_checkpoint is not None:
            self.proxy_load(epoch=load_proxy_checkpoint)
            self.proxy_step = load_proxy_checkpoint

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#
    
    def train_proxy(self, n_train_steps):
        timer = Timer()
        for step in tqdm(range(n_train_steps)):
            x, y = next(self.proxy_dataloader)
            # print(x[:4, :10])
            # print(y[:4])
            # print(self.proxy_dataset.normalizer.unnormalize(x[:4]))
            # print(self.proxy_dataset.normalizer_values.unnormalize(y[:4]))
            # print(kyle)
            # print(batch[0].shape, batch[1].shape)
            # batch = batch_to_device(batch, device=self.device)
            x = x.to(self.device)
            y = y.to(self.device)
            loss, infos = self.proxy_model.loss(x, y)
            loss.backward()
            
            self.proxy_optimizer.step()
            self.proxy_optimizer.zero_grad()
            
            if self.proxy_step % self.proxy_log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                logger.print(f'{self.proxy_step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')
                metrics = {k:v.detach().item() for k, v in infos.items()}
                metrics['proxy_steps'] = self.proxy_step
                metrics['proxy_loss'] = loss.detach().item()
                logger.log_metrics_summary(metrics, default_stats='mean')
                # wandb.log(metrics)

            self.proxy_step += 1

            if self.proxy_step % self.proxy_save_freq == 0:
                self.proxy_save()

    def train(self, n_train_steps):

        timer = Timer()
        # for step in tqdm(range(n_train_steps)):
        for step in range(n_train_steps):
            # for i in range(self.gradient_accumulate_every):
            batch = next(self.dataloader)
            traj, seid, label = batch
            label = label.unsqueeze(1)
                        
            # sid = batch_data[1][:,0].unsqueeze(1)
            # eid = batch_data[1][:,1].unsqueeze(1)
            
            # trip_len = torch.sum(traj[:,:,2]!=0, dim=1).unsqueeze(1)
            # avg_feat = torch.sum(traj[:,:,3:8], dim=1) / (trip_len+1e-6)
            # total_dist = torch.sum(traj[:,:,3], dim=1).unsqueeze(1)
            # total_time = torch.sum(traj[:,:,2], dim=1).unsqueeze(1)
            # avg_dist = avg_feat[:,0].unsqueeze(1)
            # avg_speed = avg_feat[:,1].unsqueeze(1)
            # trip_len = trip_len / self.horizon
            # total_time = total_time / 3000.
            # head = torch.cat([label, total_dist, total_time, trip_len, avg_dist, avg_speed, seid],dim=1).cuda()
            head = None
            xy = traj[:,:,:2].permute(0,2,1).cuda()
            conditions={"ctx_len": torch.zeros([])}#]*xy.shape[0]
            
            loss, infos, _ = self.model.loss(x=xy, cond=conditions)
            # batch = batch_to_device(batch, device=self.device)
            # loss, infos = self.model.loss(*batch)
            # loss = loss / self.gradient_accumulate_every
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.8f}' for key, val in infos.items()])
                logger.print(f'{self.step}: {loss:8.8f} | {infos_str} | t: {timer():8.4f}')
                metrics = {k:v.detach().item() for k, v in infos.items()}
                metrics['steps'] = self.step
                metrics['loss'] = loss.detach().item()
                logger.log_metrics_summary(metrics, default_stats='mean')
                # wandb.log(metrics)

            # if self.step == 0 and self.sample_freq:
            #     self.render_reference(self.n_reference)

            # if self.sample_freq and self.step % self.sample_freq == 0:
            #     if self.model.__class__ == diffuser.models.diffusion.GaussianInvDynDiffusion:
            #         self.inv_render_samples()
            #     elif self.model.__class__ == diffuser.models.diffusion.ActionGaussianDiffusion:
            #         pass
            #     else:
            #         self.render_samples()

            self.step += 1
            
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                self.save()
                
    def sample(self, n_samples, topk, horizon, context_length, inpainting, guidance, confidence=False):
        new_trajectories = []
        new_trajectories_confidence_score = []
        for step in tqdm(range(0, n_samples, self.batch_size)):
            batch = next(self.dataloader)
            batch = batch_to_device(batch, device=self.device)
            
            conditions = {i: to_torch(batch.trajectories[:, i], device=self.device) for i in range(context_length)}
            conditions["ctx_len"] = to_torch(np.ones(self.batch_size,), device=self.device) * context_length
           
            if inpainting:
                values = torch.ones(1, ).to(device=self.device).unsqueeze(0)
                values = values.repeat(self.batch_size, horizon-context_length)
                # values = torch.linspace(batch.trajectories[Config.horizon-context_length-1, -1], 1.0, steps=Config.horizon).to(device=device).unsqueeze(0)
            else:
                values = None
            
            if guidance:
                returns = torch.ones(1, ).to(device=self.device).unsqueeze(0)
                returns = returns.repeat(self.batch_size, 1)
                # returns = (to_torch(batch.trajectories[:, :context_length, -1].sum(axis=-1, keepdims=True), device=self.device) + horizon - context_length) / horizon
            else:
                # returns = torch.ones(1, ).to(device=self.device).unsqueeze(0) * 0.0
                # returns = returns.repeat(self.batch_size, 1)
                returns = None

            new_trajectory = self.ema_model.conditional_sample(conditions, values=values, returns=returns)
            # new_trajectory = self.ema_model.back_and_forth_sample(batch.trajectories, conditions, values=values, returns=returns)
            new_observation = self.proxy_dataset.normalizer.normalize(self.dataset.normalizer.unnormalize(new_trajectory[..., context_length:, :-1].cpu()).reshape(self.batch_size*(horizon - context_length), -1)).to(self.device)
            new_trajectory_score, new_trajectory_confidence_score = self.proxy_model(new_observation, confidence=True)
            # print(new_trajectory_score.flatten()[:10])
            new_trajectory_score = self.dataset.normalizer_values.normalize(self.proxy_dataset.unnormalize_values(new_trajectory_score.cpu())).to(self.device)
            # print(new_trajectory[:, context_length:, -1].flatten()[:10])
            # print(new_trajectory_score.flatten()[:10])
            # print(kyle)
            new_trajectory[..., context_length:, -1] = new_trajectory_score.reshape(self.batch_size, horizon-context_length)
            new_trajectory_confidence_score = new_trajectory_confidence_score.reshape(self.batch_size, horizon-context_length)
            
            new_trajectory = new_trajectory.cpu().detach()
            new_trajectory_confidence_score = new_trajectory_confidence_score.cpu().detach()
            
            new_trajectories.append(new_trajectory)
            new_trajectories_confidence_score.append(new_trajectory_confidence_score)
        new_trajectories = torch.cat(new_trajectories, dim=0)
        # print(new_trajectories[:, context_length:, -1].flatten())
        new_trajectories_confidence_score = torch.cat(new_trajectories_confidence_score, dim=0)
        if confidence:
            new_trajectories = new_trajectories[torch.argsort(new_trajectories_confidence_score.sum(axis=-1))[:topk]]
        else:
            new_trajectories = new_trajectories[torch.argsort(new_trajectories[..., context_length:, -1].sum(axis=-1))[-topk:]]
        print(new_trajectories.shape)

        optima = 1.0
        num_trajectories = self.dataset.num_trajectories + new_trajectories.shape[0]
        # print(self.dataset.points[0, :4, :10])
        # print(self.dataset.normalizer.unnormalize(new_trajectories[..., :-1])[0, :4, :10])
        # print(self.dataset.values[0, -10:])
        # print(self.dataset.normalizer_values.unnormalize(new_trajectories[..., -1])[0, -10:])
        # print(kyle)
        points = torch.cat([self.dataset.points, self.dataset.normalizer.unnormalize(new_trajectories[..., :-1])], dim=0)
        values = torch.cat([self.dataset.values, self.dataset.normalizer_values.unnormalize(new_trajectories[..., -1])], dim=0)
        
        # print(self.dataset.points[0, :10], new_trajectories[..., :-1][0, :10], self.dataset.normalizer.unnormalize(new_trajectories[..., :-1][0, :10]))
        # print(self.dataset.values[0, :10], new_trajectories[..., -1][0, :10], self.dataset.normalizer_values.unnormalize(new_trajectories[..., -1][0, :10]))
        
        pointwise_regret = optima - values
        cumulative_regret_to_go = torch.flip(torch.cumsum(torch.flip(pointwise_regret, [1]), 1), [1])
        timesteps = torch.arange(horizon).repeat(num_trajectories, 1)
        
        self.dataset.num_trajectories = num_trajectories
        self.dataset.points = points
        self.dataset.values = values
        self.dataset.pointwise_regret = pointwise_regret
        self.dataset.cumulative_rtg = cumulative_regret_to_go
        self.dataset.timesteps = timesteps
        
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=0, shuffle=True, pin_memory=True, drop_last=True,
        ))
                
    def proxy_save(self):
        data = {
            'step': self.proxy_step,
            'model': self.proxy_model.state_dict(),
        }
        savepath = os.path.join(logger.prefix, 'proxy_checkpoint')
        os.makedirs(savepath, exist_ok=True)
        # logger.save_torch(data, savepath)
        if self.save_checkpoints:
            savepath = os.path.join(savepath, f'state_{self.proxy_step}.pt')
        else:
            savepath = os.path.join(savepath, 'state.pt')
        torch.save(data, savepath)
        logger.print(f'[ utils/training ] Saved model to {savepath}')

    def save(self):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(logger.prefix, 'checkpoint')
        os.makedirs(savepath, exist_ok=True)
        # logger.save_torch(data, savepath)
        if self.save_checkpoints:
            savepath = os.path.join(savepath, f'state_{self.step}.pt')
        else:
            savepath = os.path.join(savepath, 'state.pt')
        torch.save(data, savepath)
        logger.print(f'[ utils/training ] Saved model to {savepath}')
        
    def proxy_load(self, epoch=1000):
        loadpath = os.path.join(self.bucket, logger.prefix, f'proxy_checkpoint/state_{epoch}.pt')
        # data = logger.load_torch(loadpath)
        data = torch.load(loadpath)
        self.proxy_model.load_state_dict(data['model'])

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.bucket, logger.prefix, f'checkpoint/state_{epoch}.pt')
        # data = logger.load_torch(loadpath)
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
