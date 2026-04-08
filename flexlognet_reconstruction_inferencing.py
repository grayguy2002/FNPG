# -*- coding: utf-8 -*-
"""
FlexLogNet Computational Efficiency Benchmark
Modified to compare with FNPG computational metrics
"""

import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import torch_geometric.data as gdata
from types import SimpleNamespace
import itertools
from scipy.interpolate import interp1d
from skimage.measure import block_reduce
from typing import Union, Tuple, Optional
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import softmax, unbatch
from torch_geometric.nn.inits import glorot
import math

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

#########################################################################
# Dataset Class
#########################################################################
class MyGNNDataset(gdata.Dataset):
    def __init__(self, data, sam_len, sam_num, fea_litho, norm_type, filt_size, need_maskd,
                maskc, loc_li, stap, inference_mode=False):
        gdata.Dataset.__init__(self)
        self.well_data = data
        self.sam_len = sam_len
        self.sam_num = sam_num
        self.fea_litho = fea_litho
        self.norm_type = norm_type
        self.filt_size = filt_size
        self.need_maskd = need_maskd
        self.maskc = maskc
        self.loc_li = loc_li
        self.inference_mode = inference_mode
        self.stap = stap
        self.stap_cycle = itertools.cycle(self.stap)

    def __getitem__(self, index):
        current_stap = next(self.stap_cycle)
        assert current_stap < self.loc_li[-1] - self.sam_len

        label_ori, label, mask0, sam_stap, sam_endp, well_order, well_stap, well_endp, np_mean, np_std = \
            self.label_mask0(self.well_data, current_stap)

        if self.need_maskd == True:
            maskd = self.mask_d(label, mask0)
        else:
            maskd = np.ones((label.shape[0], label.shape[1]))

        input_old = label * maskd
        input_ori = label_ori * maskd

        input = input_old.T
        input_d4 = self.resample(input, 4)
        input_d8 = self.resample(input, 8)
        input_d16 = self.resample(input, 16)

        input_3dim = input[:, np.newaxis, :]
        d4_3dim = input_d4[:, np.newaxis, :]
        d8_3dim = input_d8[:, np.newaxis, :]
        d16_3dim = input_d16[:, np.newaxis, :]

        input_4ch = np.concatenate((input_3dim, d4_3dim, d8_3dim, d16_3dim), axis=1)

        edge_index_ini, edge_ini_list, edge_index, edge_list = self.construct_edge(input_old)

        x_ori = torch.tensor(input_ori.T, dtype=torch.float)
        x = torch.tensor(input_4ch, dtype=torch.float)
        label = torch.tensor(label.T, dtype=torch.float)
        label_ori = torch.tensor(label_ori.T, dtype=torch.float)
        mask0_1 = torch.tensor(mask0.T, dtype=torch.float)
        maskd = torch.tensor(maskd.T, dtype=torch.float)
        edge_index_ini = torch.tensor(edge_index_ini, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        norm_mean = torch.tensor(np_mean.T, dtype=torch.float)
        norm_std = torch.tensor(np_std.T, dtype=torch.float)

        return gdata.Data(x=x, edge_index_ini=edge_index_ini, edge_index=edge_index, y=label,
                         mask=mask0_1, maskd=maskd, edge_ini_list=edge_ini_list, edge_list=edge_list,
                         sam_stap=sam_stap, sam_endp=sam_endp, well_order=well_order,
                         well_stap=well_stap, well_endp=well_endp,
                         norm_mean=norm_mean, norm_std=norm_std,
                         label_ori=label_ori, x_ori=x_ori)

    def __len__(self):
        return self.sam_num

    def resample(self, input, down):
        input_d = block_reduce(input, block_size=(1, down), func=np.median)
        input_ip = np.zeros((input_d.shape[0], self.sam_len))
        for i in range(input_d.shape[0]):
            input_ip[i, :] = self.interp(input_d[i, :], i, down)
        return input_ip

    def interp(self, seq, i, down):
        x = np.linspace(1, len(seq), len(seq))
        y = seq
        f = interp1d(x, y, kind='quadratic')
        xnew = np.linspace(1, len(seq), self.sam_len)
        ynew = f(xnew)
        return ynew

    def construct_edge(self, input):
        fea_n = len(self.fea_litho)
        pos = np.zeros([fea_n * (fea_n - 1), 2])
        k = 0
        for i in range(fea_n ** 2):
            if int(i / fea_n) != int(i % fea_n):
                pos[k, 1] = int(i / fea_n)
                pos[k, 0] = int(i % fea_n)
                k += 1
        pos_all = pos.copy()

        pos_all_df = pd.DataFrame(pos_all, columns=['source', 'des'])
        edge_index_list = pos_all_df.index.values
        edge_index_list = np.array(edge_index_list)
        edge_index_list = edge_index_list.T

        pos_all = pos_all.T

        a = input.sum(axis=0)
        loc = np.argwhere(a == 0)
        loc = loc.flatten()

        posdf = pd.DataFrame(pos, columns=['source', 'des'])
        for iter in range(len(loc)):
            posdf = posdf[posdf['source'] != loc[iter]].copy()

        edge_index_ini_list = posdf.index.values
        edge_index_ini_list = np.array(edge_index_ini_list)
        edge_index_ini_list = edge_index_ini_list.T

        pos = np.array(posdf)
        pos = pos.T
        return pos, edge_index_ini_list, pos_all, edge_index_list

    def depth_loc(self, a):
        b = int((a - 136.086) / 0.152)
        return b

    def label_mask0(self, df, current_stap):
        awmd = df['DEPTH_MD'].copy()
        awfea = df[self.fea_litho].copy()

        if self.inference_mode:
            stap = current_stap
        else:
            stap = np.random.randint(0, len(awfea) - self.sam_len)

        while(self.maskc[stap] == 0):
            stap = np.random.randint(0, len(awfea) - self.sam_len)
        endp = stap + self.sam_len

        i = 0
        while(self.loc_li[i] < stap):
            if stap < self.loc_li[i + 1]:
                sam_stap = self.depth_loc(awmd.iloc[stap])
                sam_endp = self.depth_loc(awmd.iloc[stap + self.sam_len])
                well_order = i
                well_stap = self.depth_loc(awmd.iloc[self.loc_li[i] + 1])
                well_endp = self.depth_loc(awmd.iloc[self.loc_li[i + 1]])
            i += 1

        label_ori = awfea.iloc[stap:endp, :].copy()
        label, np_mean, np_std = self.mean_var_norm(label_ori)

        mask0 = np.where(pd.isnull(label), 0, 1)
        label_np = np.where(pd.isnull(label), 0, label)
        label_ori = np.where(pd.isnull(label_ori), 0, label_ori)

        return label_ori, label_np, mask0, sam_stap, sam_endp, well_order, well_stap, well_endp, np_mean, np_std

    def mean_var_norm(self, label):
        df1_mean = label.mean()
        df1_mean = np.where(pd.isnull(df1_mean), 0, df1_mean)
        df1_std = label.std(ddof=0)
        df1_std = np.where(pd.isnull(df1_std), 0, df1_std)

        label_mean_var_norm = (label - df1_mean) / df1_std

        np_mean = np.array(df1_mean).reshape((1, len(self.fea_litho)))
        np_std = np.array(df1_std).reshape((1, len(self.fea_litho)))
        return label_mean_var_norm, np_mean, np_std

    def mask_d(self, label, mask0):
        maskd = np.ones((label.shape[0], label.shape[1]))
        a = maskd.shape[0]
        b = maskd.shape[1]

        m0_loc = mask0.sum(axis=0)
        loc = np.argwhere(m0_loc == 0)
        loc = loc.flatten()

        if mask0.sum() <= a * (b / 2):
            maskd = maskd
        else:
            num_maskd = 1
            maskd = self.mask_d_loc(a, b, num_maskd, maskd, loc)

        return maskd

    def mask_d_loc(self, a, b, num_maskd, maskd, loc):
        num = np.random.randint(b, size=num_maskd)
        while(set(num) <= set(loc) or self.fea_litho.index('GR') in num):
            num = np.random.randint(b, size=num_maskd)
        for i in range(num_maskd):
            maskd[:, num[i]] = 0
        return maskd

#########################################################################
# Model Architecture - Convolution Modules
#########################################################################
class GNN_conv_downsample(nn.Module):
    def __init__(self, in_ch, out_ch, onlypoolout, poolstride=2):
        super(GNN_conv_downsample, self).__init__()
        self.onlypoolout = onlypoolout
        self.Conv = nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(in_channels=out_ch, out_channels=out_ch, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.downsample = nn.Sequential(
            nn.MaxPool1d(5, stride=poolstride, padding=2)
        )

    def forward(self, x):
        out_cat = self.Conv(x)
        out = self.downsample(out_cat)
        if self.onlypoolout == True:
            return out
        else:
            return out, out_cat


class GNN_conv_upsample(nn.Module):
    def __init__(self, in_ch, out_ch, poolstride=2):
        super(GNN_conv_upsample, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=2*out_ch, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(2*out_ch),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(in_channels=2*out_ch, out_channels=2*out_ch, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(2*out_ch),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose1d(in_channels=2*out_ch, out_channels=out_ch, kernel_size=7,
                             stride=poolstride, padding=3, output_padding=(poolstride-1)),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class NoaggreGCNN(MessagePassing):
    _alpha: Optional[Tensor]

    def __init__(self, gnn_in_channels: Union[int, Tuple[int, int]], gnn_out_channels: int,
                 cnn_in_channels: int, cnn_out_channels: int, batchsize: int = 3,
                 down_up: bool = True, ifpool: bool = False, poolstride: int = 2,
                 onlypoolout: bool = True, heads: int = 1, concat: bool = False,
                 beta: bool = False, dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = False, root_weight: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(NoaggreGCNN, self).__init__(node_dim=0, **kwargs)

        self.gnn_in_channels = gnn_in_channels
        self.gnn_out_channels = gnn_out_channels
        self.cnn_in_channels = cnn_in_channels
        self.cnn_out_channels = cnn_out_channels
        self.batchsize = batchsize
        self.heads = cnn_out_channels
        self.down_up = down_up
        self.ifpool = ifpool
        self.poolstride = poolstride
        self.onlypoolout = onlypoolout
        self.device = device

        if isinstance(gnn_in_channels, int):
            gnn_in_channels = (gnn_in_channels, gnn_in_channels)

        if self.down_up == True:
            self.conv0 = GNN_conv_downsample(cnn_in_channels, cnn_out_channels, onlypoolout, poolstride).to(self.device)
            self.conv1 = GNN_conv_downsample(cnn_in_channels, cnn_out_channels, onlypoolout, poolstride).to(self.device)
            self.conv2 = GNN_conv_downsample(cnn_in_channels, cnn_out_channels, onlypoolout, poolstride).to(self.device)
            self.conv3 = GNN_conv_downsample(cnn_in_channels, cnn_out_channels, onlypoolout, poolstride).to(self.device)
        else:
            self.conv0 = GNN_conv_upsample(cnn_in_channels, cnn_out_channels, poolstride).to(self.device)
            self.conv1 = GNN_conv_upsample(cnn_in_channels, cnn_out_channels, poolstride).to(self.device)
            self.conv2 = GNN_conv_upsample(cnn_in_channels, cnn_out_channels, poolstride).to(self.device)
            self.conv3 = GNN_conv_upsample(cnn_in_channels, cnn_out_channels, poolstride).to(self.device)

    def forward(self, x: Union[Tensor, Tuple[Tensor, Tensor]], edge_index):
        CIN, COUT, GIN, GOUT = self.cnn_in_channels, self.cnn_out_channels, self.gnn_in_channels, self.gnn_out_channels

        if isinstance(x, Tensor):
            x = (x, x)

        y = torch.zeros([self.batchsize * 4, COUT, GOUT]).to(self.device)
        out_cat = torch.zeros([self.batchsize * 4, COUT, GIN]).to(self.device)

        if self.onlypoolout == True:
            y[0:self.batchsize * 4:4][:] = self.conv0(x[0][0:self.batchsize * 4:4][:])
            y[1:self.batchsize * 4:4][:] = self.conv1(x[0][1:self.batchsize * 4:4][:])
            y[2:self.batchsize * 4:4][:] = self.conv2(x[0][2:self.batchsize * 4:4][:])
            y[3:self.batchsize * 4:4][:] = self.conv3(x[0][3:self.batchsize * 4:4][:])
        else:
            y[0:self.batchsize * 4:4][:], out_cat[0:self.batchsize * 4:4][:] = self.conv0(x[0][0:self.batchsize * 4:4][:])
            y[1:self.batchsize * 4:4][:], out_cat[1:self.batchsize * 4:4][:] = self.conv1(x[0][1:self.batchsize * 4:4][:])
            y[2:self.batchsize * 4:4][:], out_cat[2:self.batchsize * 4:4][:] = self.conv2(x[0][2:self.batchsize * 4:4][:])
            y[3:self.batchsize * 4:4][:], out_cat[3:self.batchsize * 4:4][:] = self.conv3(x[0][3:self.batchsize * 4:4][:])

        return y, out_cat


class SimpleGNN(MessagePassing):
    def __init__(self, gnn_in_channels, gnn_out_channels, cnn_in_channels, cnn_out_channels,
                 batchsize: int = 32, dropout: float = 0., **kwargs):
        super(SimpleGNN, self).__init__(node_dim=0, **kwargs)
        self.gnn_in_channels = gnn_in_channels
        self.gnn_out_channels = gnn_out_channels
        self.cnn_in_channels = cnn_in_channels
        self.cnn_out_channels = cnn_out_channels
        self.batchsize = batchsize
        self.dropout = dropout

        if isinstance(self.gnn_in_channels, int):
            self.gnn_in_channels = (gnn_in_channels, gnn_in_channels)

        self.lin_transform = Linear(cnn_in_channels * self.gnn_in_channels[0],
                                    cnn_out_channels * gnn_out_channels)
        self.shift = torch.nn.Parameter(torch.Tensor(12, gnn_out_channels, gnn_out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_transform.reset_parameters()
        torch.nn.init.xavier_uniform_(self.shift)

    def forward(self, x, edge_index, edge_list):
        x = self.lin_transform(x.view(-1, self.cnn_in_channels * self.gnn_in_channels[0]))
        x = x.view(-1, self.cnn_out_channels, self.gnn_out_channels)
        return self.propagate(edge_index, x=x, edge_list=edge_list)

    def message(self, x_j, edge_list):
        out = F.dropout(x_j, p=self.dropout, training=self.training)
        shift_w = self.shift.repeat(self.batchsize, 1, 1)
        out_sum_batch = torch.zeros_like(out)

        start = 0
        for i, edges in enumerate(edge_list):
            end = start + len(edges)
            out_sum_batch[start:end] = torch.matmul(out[start:end], shift_w[i*12:(i+1)*12][edges])
            start = end

        return out_sum_batch


class upsample(nn.Module):
    def __init__(self, gnn_out_ch, cnn_in_ch, cnn_out_ch, bsize, poolstride=2):
        super(upsample, self).__init__()
        self.bsize = bsize
        self.cout = cnn_out_ch
        self.gout = gnn_out_ch

        self.conv0 = GNN_conv_upsample(cnn_in_ch, cnn_out_ch, poolstride)
        self.conv1 = GNN_conv_upsample(cnn_in_ch, cnn_out_ch, poolstride)
        self.conv2 = GNN_conv_upsample(cnn_in_ch, cnn_out_ch, poolstride)
        self.conv3 = GNN_conv_upsample(cnn_in_ch, cnn_out_ch, poolstride)

    def forward(self, x):
        y = torch.zeros([self.bsize * 4, self.cout, self.gout]).to(device)
        y[0:self.bsize * 4:4][:] = self.conv0(x[0:self.bsize * 4:4][:])
        y[1:self.bsize * 4:4][:] = self.conv1(x[1:self.bsize * 4:4][:])
        y[2:self.bsize * 4:4][:] = self.conv2(x[2:self.bsize * 4:4][:])
        y[3:self.bsize * 4:4][:] = self.conv3(x[3:self.bsize * 4:4][:])
        return y


class cat_fuse(nn.Module):
    def __init__(self, cnn_in_ch, cnn_out_ch):
        super(cat_fuse, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv1d(in_channels=cnn_in_ch, out_channels=cnn_out_ch, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(cnn_out_ch),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        y = self.Conv(x)
        return y


class MLP(torch.nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, hid_ch),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hid_ch, hid_ch),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hid_ch, out_ch),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class mean_std_head(torch.nn.Module):
    def __init__(self):
        super(mean_std_head, self).__init__()
        self.mlp1 = MLP(4*512, 4*128, 4*32)
        self.mlp2 = MLP(4*32, 4*8, 8)

    def forward(self, x, batch):
        y = torch.stack(unbatch(x, batch))
        y = torch.flatten(y, 1, -1)
        y = self.mlp1(y)
        y = self.mlp2(y)
        return y


#########################################################################
# Main Model - GraphNet_mini
#########################################################################
class GraphNet_mini(torch.nn.Module):
    def __init__(self, gnn_ch, batchsize, fea_litho, head, drop):
        super(GraphNet_mini, self).__init__()

        CNN_ch = [4, 32, 64, 128, 256, 512]
        result_ch = 1
        self.bsize = batchsize
        self.fea_n = len(fea_litho)

        self.head = mean_std_head()
        self.gd0 = NoaggreGCNN(gnn_ch[0], gnn_ch[1], CNN_ch[0], CNN_ch[1], batchsize=batchsize,
                              onlypoolout=False, down_up=True)
        self.gd1 = NoaggreGCNN(gnn_ch[1], gnn_ch[2], CNN_ch[1], CNN_ch[2], batchsize=batchsize,
                              onlypoolout=False, down_up=True)
        self.gd2 = NoaggreGCNN(gnn_ch[2], gnn_ch[3], CNN_ch[2], CNN_ch[3], batchsize=batchsize,
                              onlypoolout=False, down_up=True)

        self.u0 = upsample(gnn_ch[2], CNN_ch[3], CNN_ch[3], bsize=batchsize, poolstride=2).to(device)
        self.cat0 = cat_fuse(CNN_ch[4], CNN_ch[3]).to(device)
        self.gu0 = SimpleGNN(gnn_ch[2], gnn_ch[2], CNN_ch[3], CNN_ch[3], dropout=0.1, batchsize=batchsize)

        self.u1 = upsample(gnn_ch[1], CNN_ch[3], CNN_ch[2], bsize=batchsize, poolstride=2).to(device)
        self.cat1 = cat_fuse(CNN_ch[3], CNN_ch[2]).to(device)
        self.gu1 = SimpleGNN(gnn_ch[1], gnn_ch[1], CNN_ch[2], CNN_ch[2], dropout=0.1, batchsize=batchsize)

        self.u2 = upsample(gnn_ch[0], CNN_ch[2], CNN_ch[1], bsize=batchsize, poolstride=2).to(device)
        self.cat2 = cat_fuse(CNN_ch[2], CNN_ch[1]).to(device)
        self.gu2 = SimpleGNN(gnn_ch[0], gnn_ch[0], CNN_ch[1], CNN_ch[1], dropout=0.1, batchsize=batchsize)

        self.result = nn.Sequential(
            nn.Conv1d(CNN_ch[1], CNN_ch[1], kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(CNN_ch[1]),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(CNN_ch[1], CNN_ch[1], kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(CNN_ch[1]),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(CNN_ch[1], result_ch, kernel_size=7, stride=1, padding=3),
        )

    def forward(self, data):
        x, x_ori, edge_index_ini, edge_index = data.x, data.x_ori, data.edge_index_ini, data.edge_index
        edge_ini_list, edge_list, batch = data.edge_ini_list, data.edge_list, data.batch

        mean_std = self.head(x_ori, batch)
        out, cat0 = self.gd0(x, edge_index_ini)
        out, cat1 = self.gd1(out, edge_index_ini)
        out, cat2 = self.gd2(out, edge_index_ini)

        out = self.u0(out)
        out = torch.cat((cat2, out), dim=1)
        out = self.cat0(out)
        out = self.gu0(out, edge_index_ini, edge_ini_list)

        out = self.u1(out)
        out = torch.cat((cat1, out), dim=1)
        out = self.cat1(out)
        out = self.gu1(out, edge_index, edge_list)

        out = self.u2(out)
        out = torch.cat((cat0, out), dim=1)
        out = self.cat2(out)
        out = self.gu2(out, edge_index, edge_list)

        out = self.result(out)
        out = out.squeeze(1)

        bound = int(mean_std.size()[1] / 2)
        mean = mean_std[:, :bound]
        std = mean_std[:, bound:]

        return out, mean, std


#########################################################################
# Configuration and Model Loading
#########################################################################
# Configuration
args = SimpleNamespace(
    batch_size=1,
    sam_num=512,
    sam_len=512,
    graph_channels=[512, 256, 128, 64, 32],
    data_norm_type='',
    fea_litho=['GR', 'CALI', 'RSHA', 'RMED'],
    need_maskd=False,  # Set to False for inference
    filt_size=19,
    drop=0,
    head=1,
    inference_mode=True
)

# Model path - Using path from original script
# If running locally, update this path to your local model location
model_path = '/content/drive/MyDrive/logcompletion/model/ctra_exp/xyz_50.pt'

print("\n" + "="*60)
print("FlexLogNet Computational Efficiency Benchmark")
print("="*60)

# Load model
print(f"\nLoading FlexLogNet model from: {model_path}")
try:
    checkpoint = torch.load(model_path, map_location=device)
    model = GraphNet_mini(args.graph_channels, args.batch_size, args.fea_litho,
                         args.head, args.drop).to(device)
    model.load_state_dict(checkpoint['net'] if 'net' in checkpoint else checkpoint)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please update the model_path variable with the correct path to your FlexLogNet model.")
    exit(1)

# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
print(f"Total parameters: {total_params:,}")
print(f"Model size: {total_params * 4 / (1024**2):.2f} MB (assuming float32)")


#########################################################################
# Data Loading - UPDATE THESE PATHS
#########################################################################
# Data path - Using TRAINING set to match FNPG (not test set!)
# FNPG uses training data with well '15/9-15'
data_path = '/content/drive/MyDrive/logcompletion/data/Teamdata/standard_wells/minish_area/mini_train_standard_GCRR.csv'

print(f"\nLoading test data from: {data_path}")
try:
    dfTest = pd.read_csv(data_path, engine='python', sep=';')
    print(f"Data loaded successfully! Shape: {dfTest.shape}")
except Exception as e:
    print(f"Error loading data: {e}")
    print("Please update the data_path variable with the correct path to your test data.")
    exit(1)

# Prepare test dataset
well_names = dfTest['WELL'].unique()
print(f"Available wells: {len(well_names)}")
print(f"Well names: {list(well_names)}")

# IMPORTANT: Use the same well as FNPG for fair comparison
# FNPG uses well '15/9-15' (first well in training set)
target_well = '15/9-15'  # Change this to match FNPG's well

# Check if target well exists in the dataset
if target_well in well_names:
    print(f"Using target well for benchmarking: {target_well}")
    marked_well = dfTest[dfTest.WELL == target_well].copy()
else:
    print(f"Warning: Target well '{target_well}' not found in dataset!")
    print(f"Using first available well: {well_names[0]}")
    marked_well = dfTest[dfTest.WELL == well_names[0]].copy()
test_maskc = np.ones((marked_well.shape[0], 1))

# Create stap_list for inference
start_idx = marked_well.index[0]
end_idx = marked_well.index[-1]
stap_list_full = [i for i in range(start_idx, end_idx, args.sam_len)]
stap_list = (np.array(stap_list_full[:-1]) - start_idx).tolist()

# IMPORTANT: Limit to 34 samples to match FNPG benchmark
# Comment out the next line to test all available samples
stap_list = stap_list[:34] if len(stap_list) >= 34 else stap_list

test_loc_li = [-1]
loc = (end_idx - start_idx)
test_loc_li.append(loc)
test_maskc[loc - args.sam_len + 1: loc + 1] = 0

print(f"Number of inference samples: {len(stap_list)}")
print(f"Note: Limited to {len(stap_list)} samples for fair comparison with FNPG")

# IMPORTANT: Update sam_num to match actual number of samples
args.sam_num = len(stap_list)
print(f"Setting sam_num to {args.sam_num} (actual sample count)")

# Create dataset
inference_dataset = MyGNNDataset(marked_well, args.sam_len, args.sam_num, args.fea_litho,
                                args.data_norm_type, args.filt_size, args.need_maskd,
                                test_maskc, test_loc_li, stap=stap_list,
                                inference_mode=args.inference_mode)

inference_loader = DataLoader(inference_dataset, batch_size=args.batch_size, num_workers=0)


#########################################################################
# Benchmarking - Inference Timing and Memory Measurement
#########################################################################
print("\n" + "="*60)
print("Starting FlexLogNet Inference Benchmark...")
print("="*60)

# Warm-up run
print("\nPerforming warm-up run...")
with torch.no_grad():
    for i, data in enumerate(inference_loader):
        if i >= 3:  # Warm-up with 3 batches
            break
        data = data.to(device)
        _ = model(data)

if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

print("Warm-up complete. Starting timed inference...")

# Timed inference
batch_times = []
total_samples = 0

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

start_time = time.time()

with torch.no_grad():
    for i, data in enumerate(inference_loader):
        data = data.to(device)

        batch_start = time.time()
        output, mean, std = model(data)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        batch_end = time.time()
        batch_times.append(batch_end - batch_start)
        # Count actual number of samples (batches), not nodes
        total_samples += 1

if torch.cuda.is_available():
    torch.cuda.synchronize()

end_time = time.time()
total_time = end_time - start_time

# Memory measurement
if torch.cuda.is_available():
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # Convert to MB
else:
    peak_memory = 0

print(f"\nInference complete! Total time: {total_time:.4f} seconds")


#########################################################################
# Results Reporting
#########################################################################
# Calculate metrics - Use actual batch count instead of total_samples
actual_sample_count = len(batch_times)  # Number of batches processed
avg_batch_time = np.mean(batch_times) * 1000  # Convert to milliseconds
avg_sample_time = (total_time / actual_sample_count) * 1000  # Convert to milliseconds
throughput = actual_sample_count / total_time  # Samples per second

print("\n" + "="*60)
print("FlexLogNet Computational Efficiency Metrics")
print("="*60)
print(f"Device: {device}")
print(f"Total inference samples: {actual_sample_count}")
print(f"Total inference time: {total_time:.4f} seconds")
print(f"Average per batch time: {avg_batch_time:.2f} milliseconds")
print(f"Average per sample time: {avg_sample_time:.4f} milliseconds")
print(f"Throughput: {throughput:.2f} samples/second")
if torch.cuda.is_available():
    print(f"Peak memory usage: {peak_memory:.2f} MB")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")


#########################################################################
# Comparison Table with FNPG
#########################################################################
print("\n" + "="*60)
print("Comparison: FlexLogNet vs FNPG")
print("="*60)

# FNPG metrics (from user's provided data)
fnpg_metrics = {
    'Model': 'FNPG',
    'Total Samples': 34,
    'Total Time (s)': 0.6968,
    'Avg Time per Sample (ms)': 20.4934,
    'Throughput (samples/s)': 48.80,
    'Peak Memory (MB)': 293.19,
    'Parameters': '9.8M',
    'Device': 'NVIDIA L4'
}

# FlexLogNet metrics
flexlognet_metrics = {
    'Model': 'FlexLogNet',
    'Total Samples': actual_sample_count,
    'Total Time (s)': round(total_time, 4),
    'Avg Time per Sample (ms)': round(avg_sample_time, 4),
    'Throughput (samples/s)': round(throughput, 2),
    'Peak Memory (MB)': round(peak_memory, 2) if torch.cuda.is_available() else 'N/A',
    'Parameters': f'{total_params/1e6:.1f}M',
    'Device': str(device)
}

# Create comparison table
print("\n{:<25} {:<15} {:<15}".format("Metric", "FNPG", "FlexLogNet"))
print("-" * 60)
print("{:<25} {:<15} {:<15}".format("Total Samples", fnpg_metrics['Total Samples'],
                                     flexlognet_metrics['Total Samples']))
print("{:<25} {:<15} {:<15}".format("Total Time (s)", fnpg_metrics['Total Time (s)'],
                                     flexlognet_metrics['Total Time (s)']))
print("{:<25} {:<15} {:<15}".format("Avg Time/Sample (ms)", fnpg_metrics['Avg Time per Sample (ms)'],
                                     flexlognet_metrics['Avg Time per Sample (ms)']))
print("{:<25} {:<15} {:<15}".format("Throughput (samples/s)", fnpg_metrics['Throughput (samples/s)'],
                                     flexlognet_metrics['Throughput (samples/s)']))
print("{:<25} {:<15} {:<15}".format("Peak Memory (MB)", fnpg_metrics['Peak Memory (MB)'],
                                     flexlognet_metrics['Peak Memory (MB)']))
print("{:<25} {:<15} {:<15}".format("Parameters", fnpg_metrics['Parameters'],
                                     flexlognet_metrics['Parameters']))
print("{:<25} {:<15} {:<15}".format("Device", fnpg_metrics['Device'],
                                     flexlognet_metrics['Device']))

# Calculate speedup/slowdown
if total_samples >= fnpg_metrics['Total Samples']:
    time_ratio = avg_sample_time / fnpg_metrics['Avg Time per Sample (ms)']
    memory_ratio = peak_memory / fnpg_metrics['Peak Memory (MB)'] if torch.cuda.is_available() else 0
    param_ratio = total_params / 9.8e6

    print("\n" + "="*60)
    print("Relative Performance (FlexLogNet / FNPG)")
    print("="*60)
    print(f"Time per sample ratio: {time_ratio:.2f}x")
    if time_ratio > 1:
        print(f"  → FlexLogNet is {time_ratio:.2f}x SLOWER than FNPG")
    else:
        print(f"  → FlexLogNet is {1/time_ratio:.2f}x FASTER than FNPG")

    if torch.cuda.is_available():
        print(f"Memory usage ratio: {memory_ratio:.2f}x")
        if memory_ratio > 1:
            print(f"  → FlexLogNet uses {memory_ratio:.2f}x MORE memory than FNPG")
        else:
            print(f"  → FlexLogNet uses {1/memory_ratio:.2f}x LESS memory than FNPG")

    print(f"Parameter count ratio: {param_ratio:.2f}x")
    print(f"  → FlexLogNet has {param_ratio:.2f}x MORE parameters than FNPG")

print("\n" + "="*60)
print("Benchmark Complete!")
print("="*60)

