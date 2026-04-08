# -*- coding: utf-8 -*-
"""
FNPG kmax Sensitivity Evaluation Script
评估不同 kmax 值的模型性能，计算完整的量化指标

使用方法:
    python kmax_evaluation.py

输出指标:
    - MSE (Mean Squared Error)
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - R² (Coefficient of Determination)
    - Pearson Correlation
    - MAPE (Mean Absolute Percentage Error)
    - Max Error
"""

import torch
import glob
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import reduce
import operator
from typing import Union, Tuple
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# 配置参数
# ============================================================================
KMAX_VALUES = [2, 4, 8, 16, 32]  # 要评估的 kmax 值
BASE_PATH = '/content/drive/MyDrive/logcompletion/model/kmax_sensitivity'
DATA_PATH_X = '/content/drive/MyDrive/logcompletion/data/Teamdata/complete_data_for_PIModel/train/Train_x_all_complete_*.pt'
DATA_PATH_Y = '/content/drive/MyDrive/logcompletion/data/Teamdata/complete_data_for_PIModel/train/Train_y_all_complete_*.pt'
DATA_PATH_TARGET_GCRR = '/content/drive/MyDrive/logcompletion/data/Teamdata/complete_data_for_PIModel/target_10/Target_GCRR_*.pt'
DATA_PATH_TARGET_GRSB = '/content/drive/MyDrive/logcompletion/data/Teamdata/complete_data_for_PIModel/target_10/Target_GRSB_*.pt'
OUTPUT_DIR = '/content/drive/MyDrive/logcompletion/metrics'

BATCH_SIZE = 2  # 推理时的 batch size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# 数据类定义
# ============================================================================
class Data:
    def __init__(self, input_sam, label):
        self.x = input_sam
        self.y = label
    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        return self

class my_dataset(Dataset):
    def __init__(self, input, target):
        self.input = input
        self.target = target

    def __getitem__(self, idx):
        input_sam = self.input[idx]
        label = self.target[idx]

        if not isinstance(input_sam, torch.Tensor):
            input_sam = torch.tensor(input_sam, dtype=torch.float32)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.float32)

        return Data(input_sam=input_sam, label=label)

    def __len__(self):
        return len(self.input)

def custom_collate(batch):
    input_sam = torch.stack([item.x for item in batch])
    label = torch.stack([item.y for item in batch])
    return Data(input_sam=input_sam, label=label)

# ============================================================================
# 模型定义
# ============================================================================
class conv_down_conv_obj(nn.Module):
    def __init__(self, in_ch, out_ch, onlypoolout, poolstride=2):
        super(conv_down_conv_obj, self).__init__()
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
        if self.onlypoolout:
            return out
        else:
            return out, out_cat

class conv_up_conv_obj(nn.Module):
    def __init__(self, in_ch, out_ch, poolstride=2):
        super(conv_up_conv_obj, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=2*out_ch, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(2*out_ch),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(in_channels=2*out_ch, out_channels=2*out_ch, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(2*out_ch),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose1d(in_channels=2*out_ch, out_channels=out_ch, kernel_size=7, stride=poolstride, padding=3, output_padding=(poolstride-1)),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        return self.layers(x)

class conv_obj(nn.Module):
    def __init__(self, _in_channels: Union[int, Tuple[int, int]], _out_channels: int,
                 cnn_in_channels: int, cnn_out_channels: int, batchsize: int = 3,
                 down_up: bool = True, poolstride: int = 2, onlypoolout: bool = True):
        super(conv_obj, self).__init__()
        self._in_channels = _in_channels
        self._out_channels = _out_channels
        self.cnn_in_channels = cnn_in_channels
        self.cnn_out_channels = cnn_out_channels
        self.batchsize = batchsize
        self.down_up = down_up
        self.poolstride = poolstride
        self.onlypoolout = onlypoolout

        if isinstance(_in_channels, int):
            _in_channels = (_in_channels, _in_channels)

        self.device = DEVICE

        if self.down_up:
            self.conv0 = conv_down_conv_obj(cnn_in_channels, cnn_out_channels, onlypoolout, poolstride).to(self.device)
            self.conv1 = conv_down_conv_obj(cnn_in_channels, cnn_out_channels, onlypoolout, poolstride).to(self.device)
            self.conv2 = conv_down_conv_obj(cnn_in_channels, cnn_out_channels, onlypoolout, poolstride).to(self.device)
            self.conv3 = conv_down_conv_obj(cnn_in_channels, cnn_out_channels, onlypoolout, poolstride).to(self.device)
        else:
            self.conv0 = conv_up_conv_obj(cnn_in_channels, cnn_out_channels, poolstride).to(self.device)
            self.conv1 = conv_up_conv_obj(cnn_in_channels, cnn_out_channels, poolstride).to(self.device)
            self.conv2 = conv_up_conv_obj(cnn_in_channels, cnn_out_channels, poolstride).to(self.device)
            self.conv3 = conv_up_conv_obj(cnn_in_channels, cnn_out_channels, poolstride).to(self.device)

    def forward(self, x):
        CIN, COUT, GIN, GOUT = self.cnn_in_channels, self.cnn_out_channels, self._in_channels, self._out_channels
        y = torch.zeros([self.batchsize, COUT, GOUT]).to(self.device)
        out_cat = torch.zeros([self.batchsize, COUT, GIN]).to(self.device)

        if self.onlypoolout:
            y[0:self.batchsize * 4:4][:] = self.conv0(x[0:self.batchsize * 4:4][:])
            y[1:self.batchsize * 4:4][:] = self.conv1(x[1:self.batchsize * 4:4][:])
            y[2:self.batchsize * 4:4][:] = self.conv2(x[2:self.batchsize * 4:4][:])
            y[3:self.batchsize * 4:4][:] = self.conv3(x[3:self.batchsize * 4:4][:])
        else:
            y[0:self.batchsize * 4:4][:], out_cat[0:self.batchsize * 4:4][:] = self.conv0(x[0:self.batchsize * 4:4][:])
            y[1:self.batchsize * 4:4][:], out_cat[1:self.batchsize * 4:4][:] = self.conv1(x[1:self.batchsize * 4:4][:])
            y[2:self.batchsize * 4:4][:], out_cat[2:self.batchsize * 4:4][:] = self.conv2(x[2:self.batchsize * 4:4][:])
            y[3:self.batchsize * 4:4][:], out_cat[3:self.batchsize * 4:4][:] = self.conv3(x[3:self.batchsize * 4:4][:])

        return y, out_cat

class upsample(nn.Module):
    def __init__(self, gnn_out_ch, cnn_in_ch, cnn_out_ch, bsize, poolstride=2):
        super(upsample, self).__init__()
        self.bsize = bsize
        self.cout = cnn_out_ch
        self.gout = gnn_out_ch

        self.conv0 = conv_up_conv_obj(cnn_in_ch, cnn_out_ch, poolstride)
        self.conv1 = conv_up_conv_obj(cnn_in_ch, cnn_out_ch, poolstride)
        self.conv2 = conv_up_conv_obj(cnn_in_ch, cnn_out_ch, poolstride)
        self.conv3 = conv_up_conv_obj(cnn_in_ch, cnn_out_ch, poolstride)

    def forward(self, x):
        y = torch.zeros([self.bsize, self.cout, self.gout]).to(DEVICE)
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
        return self.Conv(x)

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-1]).to(x.device)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1.to(x.device))
        out_ft[:, :, -self.modes1:] = self.compl_mul1d(x_ft[:, :, -self.modes1:], self.weights2.to(x.device))
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights3.to(x.device))
        out_ft[:, :, -self.modes1:] = self.compl_mul1d(x_ft[:, :, -self.modes1:], self.weights4.to(x.device))
        x = torch.fft.irfftn(out_ft, s=(x.size(-1)))
        return x

class MLP_(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP_, self).__init__()
        self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FourierFeatureExtractor(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, mid_channels=8, modes1=8):
        super(FourierFeatureExtractor, self).__init__()
        self.modes1 = modes1
        self.fconv1 = SpectralConv1d(in_channels, mid_channels, modes1)
        self.fconv2 = SpectralConv1d(mid_channels, mid_channels, modes1)
        self.fconv3 = SpectralConv1d(mid_channels, mid_channels, modes1)
        self.mlp1 = MLP_(mid_channels, mid_channels, mid_channels)
        self.mlp2 = MLP_(mid_channels, mid_channels, mid_channels)
        self.mlp3 = MLP_(mid_channels, mid_channels, mid_channels)
        self.w1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.w2 = nn.Conv1d(mid_channels, mid_channels, 1)
        self.w3 = nn.Conv1d(mid_channels, mid_channels, 1)
        self.out_proj = MLP_(mid_channels, out_channels, mid_channels)

    def forward(self, x):
        x_f = self.fconv1(x)
        x_f = self.mlp1(x_f)
        x_c = self.w1(x)
        x = x_f + x_c
        x = F.gelu(x)

        x_f = self.fconv2(x)
        x_f = self.mlp2(x_f)
        x_c = self.w2(x)
        x = x_f + x_c
        x = F.gelu(x)

        x_f = self.fconv3(x)
        x_f = self.mlp3(x_f)
        x_c = self.w3(x)
        x = x_f + x_c
        x = F.gelu(x)

        x = self.out_proj(x)
        return x

class FNPG(torch.nn.Module):
    def __init__(self, gnn_ch, batchsize, fea_litho, head, drop, modes1=16):
        super(FNPG, self).__init__()
        CNN_ch = [4, 32, 64, 128, 256, 512]
        result_ch = 5

        self.bsize = batchsize
        self.fea_n = len(fea_litho)
        self.modes1 = modes1

        self.gd0 = conv_obj(gnn_ch[0], gnn_ch[1], CNN_ch[0], CNN_ch[1], batchsize=batchsize, onlypoolout=False, down_up=True)
        self.gd1 = conv_obj(gnn_ch[1], gnn_ch[2], CNN_ch[1], CNN_ch[2], batchsize=batchsize, onlypoolout=False, down_up=True)
        self.gd2 = conv_obj(gnn_ch[2], gnn_ch[3], CNN_ch[2], CNN_ch[3], batchsize=batchsize, onlypoolout=False, down_up=True)

        self.FFE_up1 = FourierFeatureExtractor(CNN_ch[1], CNN_ch[1], mid_channels=32, modes1=modes1)
        self.FFE_up2 = FourierFeatureExtractor(CNN_ch[2], CNN_ch[2], mid_channels=32, modes1=modes1)
        self.FFE_up3 = FourierFeatureExtractor(CNN_ch[3], CNN_ch[3], mid_channels=32, modes1=modes1)

        self.u0 = upsample(gnn_ch[2], CNN_ch[3], CNN_ch[3], bsize=batchsize, poolstride=2).to(DEVICE)
        self.cat0 = cat_fuse(CNN_ch[4], CNN_ch[3]).to(DEVICE)

        self.u1 = upsample(gnn_ch[1], CNN_ch[3], CNN_ch[2], bsize=batchsize, poolstride=2).to(DEVICE)
        self.cat1 = cat_fuse(CNN_ch[3], CNN_ch[2]).to(DEVICE)

        self.u2 = upsample(gnn_ch[0], CNN_ch[2], CNN_ch[1], bsize=batchsize, poolstride=2).to(DEVICE)
        self.cat2 = cat_fuse(CNN_ch[2], CNN_ch[1]).to(DEVICE)

        self.FFE_down1 = FourierFeatureExtractor(CNN_ch[1], CNN_ch[1], mid_channels=32, modes1=modes1)
        self.FFE_down2 = FourierFeatureExtractor(CNN_ch[2], CNN_ch[2], mid_channels=32, modes1=modes1)
        self.FFE_down3 = FourierFeatureExtractor(CNN_ch[3], CNN_ch[3], mid_channels=32, modes1=modes1)

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
        x = data.x
        out, cat0 = self.gd0(x)
        out = self.FFE_up1(out)
        out, cat1 = self.gd1(out)
        out = self.FFE_up2(out)
        out, cat2 = self.gd2(out)
        out = self.FFE_up3(out)
        out = self.u0(out)
        out = torch.cat((cat2, out), dim=1)
        out = self.cat0(out)
        out = self.FFE_down3(out)
        out = self.u1(out)
        out = torch.cat((cat1, out), dim=1)
        out = self.cat1(out)
        out = self.FFE_down2(out)
        out = self.u2(out)
        out = torch.cat((cat0, out), dim=1)
        out = self.cat2(out)
        out = self.FFE_down1(out)
        out = self.result(out)
        out = out.squeeze(1)
        return out

# ============================================================================
# 工具函数
# ============================================================================
def count_params(model):
    """计算模型参数量"""
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size() + (2,) if p.is_complex() else p.size()))
    return c

def load_checkpoint(model, path):
    """加载模型检查点"""
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint.get('epoch', 0), checkpoint.get('best_val_loss', 0)

def compute_metrics(predictions, targets):
    """
    计算所有评估指标

    Args:
        predictions: 预测值 (numpy array)
        targets: 目标值 (numpy array)

    Returns:
        dict: 包含所有指标的字典
    """
    predictions = predictions.flatten()
    targets = targets.flatten()

    # MSE
    mse = np.mean((predictions - targets) ** 2)

    # RMSE
    rmse = np.sqrt(mse)

    # MAE
    mae = np.mean(np.abs(predictions - targets))

    # R² (Coefficient of Determination)
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Pearson Correlation
    if np.std(predictions) > 0 and np.std(targets) > 0:
        pearson_r, pearson_p = stats.pearsonr(predictions, targets)
    else:
        pearson_r, pearson_p = 0, 1

    # MAPE (Mean Absolute Percentage Error) - 避免除零
    mask = targets != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
    else:
        mape = np.nan

    # Max Error
    max_error = np.max(np.abs(predictions - targets))

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Pearson_R': pearson_r,
        'Pearson_P': pearson_p,
        'MAPE': mape,
        'Max_Error': max_error
    }

def compute_per_channel_metrics(predictions, targets, channel_names):
    """
    计算每个通道的评估指标

    Args:
        predictions: 预测值 (N, C, L)
        targets: 目标值 (N, C, L)
        channel_names: 通道名称列表

    Returns:
        DataFrame: 每个通道的指标
    """
    n_channels = predictions.shape[1]
    results = []

    for i in range(n_channels):
        pred_channel = predictions[:, i, :].flatten()
        target_channel = targets[:, i, :].flatten()

        metrics = compute_metrics(pred_channel, target_channel)
        metrics['Channel'] = channel_names[i] if i < len(channel_names) else f'Channel_{i}'
        results.append(metrics)

    return pd.DataFrame(results)

# ============================================================================
# 主评估函数
# ============================================================================
def load_data():
    """加载测试数据"""
    print("Loading data...")

    # 加载数据文件
    y_files = sorted(glob.glob(DATA_PATH_Y))
    target_files_gcrr = sorted(glob.glob(DATA_PATH_TARGET_GCRR))
    target_files_grsb = sorted(glob.glob(DATA_PATH_TARGET_GRSB))

    tensor_list_y = []
    tensor_list_target = []
    tensor_list_target_2 = []

    for file in y_files:
        tensor = torch.load(file, map_location=DEVICE)
        tensor_list_y.append(tensor)

    for file in target_files_gcrr:
        tensor = torch.load(file, map_location=DEVICE)
        tensor_list_target.append(tensor)

    for file in target_files_grsb:
        tensor = torch.load(file, map_location=DEVICE)
        tensor_list_target_2.append(tensor)

    # 使用第一口井作为测试 (15/9-15)
    well_y_pt = tensor_list_y[0].clone()
    well_list_target_ = torch.cat((tensor_list_target[0][:, 1:, :].clone(),
                                    tensor_list_target_2[0][:, 1:3, :].clone()), dim=1)
    well_target_pt = well_list_target_.clone()

    print(f"Test data shape: input={well_y_pt.shape}, target={well_target_pt.shape}")

    return well_y_pt, well_target_pt

def evaluate_model(kmax, well_y_pt, well_target_pt):
    """
    评估指定 kmax 的模型

    Args:
        kmax: 傅里叶模式数
        well_y_pt: 输入数据
        well_target_pt: 目标数据

    Returns:
        dict: 评估结果
    """
    print(f"\n{'='*60}")
    print(f"Evaluating kmax={kmax}")
    print(f"{'='*60}")

    # 模型路径 - 所有 kmax 使用统一格式
    model_path = f'{BASE_PATH}/kmax_{kmax}/FNPG_kmax{kmax}_best.pt'

    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}")
        return None

    # 创建模型
    graph_channels = [512, 256, 128, 64, 32]
    fea_litho = ['GR', 'RHOB', 'NPHI', 'DTC']

    model = FNPG(graph_channels, BATCH_SIZE, fea_litho, head=1, drop=0, modes1=kmax).to(DEVICE)

    # 加载模型权重
    model, epoch, val_loss = load_checkpoint(model, model_path)
    print(f"Loaded model from epoch {epoch}, val_loss={val_loss:.6f}")

    # 计算参数量
    total_params = count_params(model)
    print(f"Total parameters: {total_params:,}")

    # 创建数据加载器
    infer_dataset = my_dataset(well_y_pt, well_target_pt)
    infer_loader = DataLoader(
        infer_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=False,
        drop_last=True,
        collate_fn=custom_collate
    )

    # 执行推理
    model.eval()
    predictions_list = []
    targets_list = []

    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(infer_loader):
            batch.x = batch.x.view(-1, 4, 512)
            output = model(batch)
            predictions_list.append(output.cpu())
            targets_list.append(batch.y.cpu())

    # 合并结果
    predictions = torch.cat(predictions_list, dim=0).numpy()
    targets = torch.cat(targets_list, dim=0).numpy()

    # 计算整体指标
    overall_metrics = compute_metrics(predictions, targets)

    # 计算每个通道的指标
    channel_names = ['CALI', 'RSHA', 'RMED', 'RDEP', 'SP']
    per_channel_metrics = compute_per_channel_metrics(predictions, targets, channel_names)

    # 加载 test loss（如果存在）
    loss_file = f'{BASE_PATH}/kmax_{kmax}/loss_kmax{kmax}_test.npy'

    test_loss = None
    if os.path.exists(loss_file):
        losses = np.load(loss_file)
        test_loss = losses[-1] if len(losses) > 0 else None
        best_loss = losses.min() if len(losses) > 0 else None
        best_epoch = losses.argmin() + 1 if len(losses) > 0 else None
    else:
        best_loss = None
        best_epoch = None

    result = {
        'kmax': kmax,
        'total_params': total_params,
        'test_loss': test_loss,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'overall_metrics': overall_metrics,
        'per_channel_metrics': per_channel_metrics,
        'predictions': predictions,
        'targets': targets
    }

    return result

def print_results_table(results):
    """打印结果汇总表格"""
    print("\n" + "=" * 100)
    print("kmax Ablation Study - Complete Metrics Summary")
    print("=" * 100)

    # 表头
    header = f"{'kmax':<6} {'Params':<12} {'Test Loss':<12} {'MSE':<12} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'Pearson R':<10}"
    print(header)
    print("-" * 100)

    # 数据行
    for r in results:
        if r is None:
            continue
        m = r['overall_metrics']
        row = f"{r['kmax']:<6} {r['total_params']:<12,} "
        row += f"{r['test_loss']:<12.6f} " if r['test_loss'] else f"{'N/A':<12} "
        row += f"{m['MSE']:<12.6f} {m['RMSE']:<10.6f} {m['MAE']:<10.6f} {m['R2']:<10.6f} {m['Pearson_R']:<10.6f}"
        print(row)

    print("=" * 100)

    # 比较分析
    print("\nPerformance Comparisons (relative to kmax=16):")
    print("-" * 60)

    kmax_16_result = next((r for r in results if r and r['kmax'] == 16), None)
    if kmax_16_result:
        base_mse = kmax_16_result['overall_metrics']['MSE']
        base_r2 = kmax_16_result['overall_metrics']['R2']

        for r in results:
            if r is None or r['kmax'] == 16:
                continue
            mse_ratio = r['overall_metrics']['MSE'] / base_mse if base_mse > 0 else 0
            r2_diff = r['overall_metrics']['R2'] - base_r2
            print(f"kmax={r['kmax']:2} vs kmax=16: MSE ratio={mse_ratio:.4f}x, R² diff={r2_diff:+.6f}")

def save_results(results, output_dir):
    """保存结果到文件"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存汇总表格
    summary_data = []
    for r in results:
        if r is None:
            continue
        m = r['overall_metrics']
        summary_data.append({
            'kmax': r['kmax'],
            'total_params': r['total_params'],
            'test_loss': r['test_loss'],
            'best_loss': r['best_loss'],
            'best_epoch': r['best_epoch'],
            'MSE': m['MSE'],
            'RMSE': m['RMSE'],
            'MAE': m['MAE'],
            'R2': m['R2'],
            'Pearson_R': m['Pearson_R'],
            'Pearson_P': m['Pearson_P'],
            'MAPE': m['MAPE'],
            'Max_Error': m['Max_Error']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'kmax_evaluation_summary.csv'), index=False)
    print(f"\nSummary saved to: {os.path.join(output_dir, 'kmax_evaluation_summary.csv')}")

    # 保存每个通道的指标
    for r in results:
        if r is None:
            continue
        r['per_channel_metrics'].to_csv(
            os.path.join(output_dir, f'kmax{r["kmax"]}_per_channel_metrics.csv'),
            index=False
        )

    print(f"Per-channel metrics saved to: {output_dir}")

def plot_comparison(results, output_dir):
    """绘制比较图"""
    os.makedirs(output_dir, exist_ok=True)

    # 提取数据
    kmax_vals = [r['kmax'] for r in results if r]
    mse_vals = [r['overall_metrics']['MSE'] for r in results if r]
    rmse_vals = [r['overall_metrics']['RMSE'] for r in results if r]
    mae_vals = [r['overall_metrics']['MAE'] for r in results if r]
    r2_vals = [r['overall_metrics']['R2'] for r in results if r]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # MSE vs kmax
    axes[0, 0].bar(range(len(kmax_vals)), mse_vals, color='steelblue', edgecolor='black')
    axes[0, 0].set_xticks(range(len(kmax_vals)))
    axes[0, 0].set_xticklabels(kmax_vals)
    axes[0, 0].set_xlabel('kmax')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_title('MSE vs kmax')
    axes[0, 0].grid(True, alpha=0.3)

    # R² vs kmax
    axes[0, 1].bar(range(len(kmax_vals)), r2_vals, color='forestgreen', edgecolor='black')
    axes[0, 1].set_xticks(range(len(kmax_vals)))
    axes[0, 1].set_xticklabels(kmax_vals)
    axes[0, 1].set_xlabel('kmax')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].set_title('R² vs kmax')
    axes[0, 1].grid(True, alpha=0.3)

    # RMSE vs kmax
    axes[1, 0].bar(range(len(kmax_vals)), rmse_vals, color='coral', edgecolor='black')
    axes[1, 0].set_xticks(range(len(kmax_vals)))
    axes[1, 0].set_xticklabels(kmax_vals)
    axes[1, 0].set_xlabel('kmax')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].set_title('RMSE vs kmax')
    axes[1, 0].grid(True, alpha=0.3)

    # MAE vs kmax
    axes[1, 1].bar(range(len(kmax_vals)), mae_vals, color='orchid', edgecolor='black')
    axes[1, 1].set_xticks(range(len(kmax_vals)))
    axes[1, 1].set_xticklabels(kmax_vals)
    axes[1, 1].set_xlabel('kmax')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].set_title('MAE vs kmax')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('kmax Ablation Study Results', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'kmax_ablation_comparison.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_path}")
    plt.show()

# ============================================================================
# 主程序
# ============================================================================
def main():
    print("=" * 70)
    print("FNPG kmax Sensitivity Evaluation")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"kmax values to evaluate: {KMAX_VALUES}")
    print("=" * 70)

    # 加载数据
    well_y_pt, well_target_pt = load_data()

    # 评估每个 kmax 值
    results = []
    for kmax in KMAX_VALUES:
        result = evaluate_model(kmax, well_y_pt, well_target_pt)
        results.append(result)

    # 打印结果表格
    print_results_table(results)

    # 保存结果
    save_results(results, OUTPUT_DIR)

    # 绘制比较图
    plot_comparison(results, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
