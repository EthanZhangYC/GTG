import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm
from types import SimpleNamespace
from utils.Traj_UNet_ori import *
from utils.config_ori import args
from utils.utils import *
from torch.utils.data import DataLoader


temp = {}
for k, v in args.items():
    temp[k] = SimpleNamespace(**v)

config = SimpleNamespace(**temp)

config.training.batch_size = batchsize = 2
unet = Guide_UNet(config).cuda()
# # load the model
# unet.load_state_dict(torch.load('./model.pt'))


n_steps = config.diffusion.num_diffusion_timesteps
beta = torch.linspace(config.diffusion.beta_start,
                          config.diffusion.beta_end, n_steps).cuda()
alpha = 1. - beta
alpha_bar = torch.cumprod(alpha, dim=0)
lr = 2e-4  # Explore this - might want it lower when training on the full dataset

eta=0.0
timesteps=100
skip = n_steps // timesteps
seq = range(0, n_steps, skip)

# # load head information for guide trajectory generation
# batchsize = 500
# head = np.load('heads.npy',
#                    allow_pickle=True)
# head = torch.from_numpy(head).float()
# dataloader = DataLoader(head, batch_size=batchsize, shuffle=True, num_workers=4)





# head = np.array([[0.0000e+00, 1.1301e-02, 3.2167e-01, 1.0000e+00, 5.6503e-05, 1.1917e-02, 1.6700e+02, 1.6700e+02],[0.0000e+00, 8.9842e-03, 1.3333e-01, 1.0000e+00, 4.4921e-05, 2.2534e-02, 1.2100e+02, 1.2100e+02]])
# head = np.array([[1.0000e+00, 2.8011e-02, 1.3333e-01, 1.0000e+00, 1.4006e-04, 7.0259e-02, 1.2100e+02, 1.2100e+02],[0.0000e+00, 8.9842e-03, 1.3333e-01, 1.0000e+00, 4.4921e-05, 2.2534e-02, 1.2100e+02, 1.2100e+02]])
# head = np.array([[2.0000e+00, 6.2662e-02, 1.4033e-01, 1.0000e+00, 3.1331e-04, 1.5628e-01, 1.2100e+02, 1.2100e+02],[0.0000e+00, 8.9842e-03, 1.3333e-01, 1.0000e+00, 4.4921e-05, 2.2534e-02, 1.2100e+02, 1.2100e+02]])
# head = np.array([[3.0000e+00, 3.2609e-02, 1.7300e-01, 1.0000e+00, 1.6305e-04, 7.8854e-02, 1.2100e+02, 1.2100e+02],[0.0000e+00, 8.9842e-03, 1.3333e-01, 1.0000e+00, 4.4921e-05, 2.2534e-02, 1.2100e+02, 1.2100e+02]])



model_dir_list=[
    # './model.pt',
    "/home/yichen/GTG/results/1025_ori_len200_bs128/checkpoint/state_10000.pt",

] 
filename='1027mtl_traj_speed.png'








# lengths=200
# head = np.load('heads.npy', allow_pickle=True)
# head = head[:2]


print(filename)
x0 = torch.randn(batchsize, 2, config.data.traj_length).cuda()
head = torch.from_numpy(head).float().cuda()



Gen_traj = []
Gen_head = []
# for i in tqdm(range(1)):
#     # head = next(iter(dataloader))
#     # lengths = head[:, 3]
#     # lengths = lengths * len_std + len_mean
#     # lengths = lengths.int()
#     # tes = head[:,:6].numpy()
#     # Gen_head.extend((tes*hstd+hmean))
#     # head = head.cuda()
#     # Start with random noise
#     x = torch.randn(batchsize, 2, config.data.traj_length).cuda()



            
            

for model_dir in model_dir_list:
    ckpt_dir = model_dir
    print(ckpt_dir)
    unet.load_state_dict(torch.load(ckpt_dir), strict=False)
    
    # lengths = head[:, 3].cpu()
    # lengths = lengths * 200
    # lengths = lengths.int()
    # # lengths = head[:, 3].cpu()
    # # lengths = lengths * len_std + len_mean
    # # lengths = lengths.int()
    
    
    batch = next(self.dataloader)
    traj, seid, label = batch
    traj = traj.permute(0,2,1)
    # label = label.unsqueeze(1)

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
    traj = traj.cuda()
    loss, infos, x_recon = self.model.loss(x=traj, cond=head)
            
            

    ims = []
    n = x0.size(0)
    x = x0
    seq_next = [-1] + list(seq[:-1])
    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        with torch.no_grad():
            pred_noise = unet(x, t, head)
            x = p_xt(x, pred_noise, t, next_t, beta, eta)
            if i % 10 == 0:
                ims.append(x.cpu().squeeze(0))
    trajs = ims[-1].cpu().numpy()
    trajs = trajs[:,:2,:]
    # resample the trajectory length
    # for j in range(batchsize):
    j=0
    new_traj = resample_trajectory(trajs[j].T, lengths[j])
    # new_traj = new_traj * std + mean
    if input_speed:
        lat_min,lat_max = (0.0, 49.95356703911097)
        lon_min,lon_max = (0.0, 49.95356703911097)
        new_traj[:,0] = new_traj[:,0] * (lat_max-lat_min) + lat_min
        new_traj[:,1] = new_traj[:,1] * (lon_max-lon_min) + lon_min        
        # start_pt = [36,72]
        tmp_x = 36
        tmp_y = 72
        new_new_traj = []
        new_new_traj.append((tmp_x,tmp_y))
        for i in range(len(new_traj)):
            tmp_x += new_traj[i,0]
            tmp_y += new_traj[i,1]
            new_new_traj.append((tmp_x,tmp_y))
        # pdb.set_trace()
        new_traj = np.array(new_new_traj)
    else:
        lat_min,lat_max = (18.249901, 55.975593)
        lon_min,lon_max = (-122.3315333, 126.998528)
        new_traj[:,0] = new_traj[:,0] * (lat_max-lat_min) + lat_min
        new_traj[:,1] = new_traj[:,1] * (lon_max-lon_min) + lon_min
    # print(new_traj)
    Gen_traj.append(new_traj)


# try:
fig = plt.figure(figsize=(12,12))
for i in range(len(Gen_traj)):
    traj=Gen_traj[i]
    ax1 = fig.add_subplot(331+i)  
    ax1.plot(traj[:,0],traj[:,1],color='blue',alpha=0.1)
# except:
#     pdb.set_trace()
plt.tight_layout()
plt.savefig(filename)
plt.show()

# plt.figure(figsize=(8,8))
# for i in range(len(Gen_traj)):
#     traj=Gen_traj[i]
#     plt.plot(traj[:,0],traj[:,1],color='blue',alpha=0.1)
# plt.tight_layout()
# plt.savefig('Chengdu_traj.png')
# plt.show()