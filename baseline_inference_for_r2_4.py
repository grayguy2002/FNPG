"""
Baseline模型推理脚本 - 用于R2-4统计显著性检验

功能：
1. 加载训练好的ResNet和LSTM模型
2. 在测试井15/9-15和15/9-17上运行推理
3. 计算每个切片的样本级MSE
4. 输出格式化数据，用于配对t检验

日期：2026-01-19
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from scipy import stats

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ============================================================================
# 模型定义部分
# ============================================================================

class BasicBlock(nn.Module):
    """ResNet基础块 - 从fnpg_train.py复制"""
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)  # Add shortcut
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet模型 - 从fnpg_train.py复制"""
    def __init__(self, block, num_blocks, input_channels=4, output_channels=5):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1])
        self.layer3 = self._make_layer(block, 256, num_blocks[2])
        self.layer4 = self._make_layer(block, 512, num_blocks[3])

        # Final output layer
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, output_channels)

    def _make_layer(self, block, out_channels, blocks):
        layers = []
        layers.append(block(self.in_channels, out_channels))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.x  # 提取tensor（模型期望接收Data对象）
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)

        return x.unsqueeze(2).repeat(1, 1, 512)  # Expand to (batch_size, 5, 512)


class LSTMModel(nn.Module):
    """Bi-LSTM模型 - 从fnpg_train.py复制"""
    def __init__(self, input_size=4, hidden_size=256, num_layers=3, output_size=5):
        super(LSTMModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Fully connected layers for output
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # x shape: (batch_size, 4, 512)
        x = x.x  # 提取tensor（模型期望接收Data对象）
        # Reshape input to (batch_size, seq_len, input_size)
        x = x.permute(0, 2, 1)  # Now shape is (batch_size, 512, 4)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Apply fully connected layers to each time step
        output = self.fc(lstm_out)

        # Reshape output to match desired shape (batch_size, 5, 512)
        output = output.permute(0, 2, 1)

        return output


# ============================================================================
# 数据加载和辅助函数
# ============================================================================

def load_test_data():
    """加载测试数据"""
    print("\n加载测试数据...")

    # 数据路径（与fnpg_generation.py保持一致）
    STAGE = 'GCRR'
    STAGE_2 = 'GRSB'

    # 使用glob加载所有训练数据文件
    file_pattern_y = '/content/drive/MyDrive/logcompletion/data/Teamdata/complete_data_for_PIModel/train/Train_y_all_complete_*.pt'
    file_pattern_target = f'/content/drive/MyDrive/logcompletion/data/Teamdata/complete_data_for_PIModel/target_10/Target_{STAGE}_*.pt'
    file_pattern_target_2 = f'/content/drive/MyDrive/logcompletion/data/Teamdata/complete_data_for_PIModel/target_10/Target_{STAGE_2}_*.pt'

    import glob
    y_files = glob.glob(file_pattern_y)
    target_files = glob.glob(file_pattern_target)
    target_files_2 = glob.glob(file_pattern_target_2)

    # 排序确保顺序正确
    y_files.sort()
    target_files.sort()
    target_files_2.sort()

    print(f"  找到 {len(y_files)} 个输入文件")
    print(f"  找到 {len(target_files)} 个目标文件(GCRR)")
    print(f"  找到 {len(target_files_2)} 个目标文件(GRSB)")

    # 加载所有文件
    tensor_list_y = []
    tensor_list_target = []
    tensor_list_target_2 = []

    for file in y_files:
        tensor = torch.load(file, map_location=device)
        tensor_list_y.append(tensor)

    for file in target_files:
        tensor = torch.load(file, map_location=device)
        tensor_list_target.append(tensor)

    for file in target_files_2:
        tensor = torch.load(file, map_location=device)
        tensor_list_target_2.append(tensor)

    print(f"  加载完成！总共 {len(tensor_list_y)} 口井")
    print(f"  井0形状: {tensor_list_y[0].shape}")
    if len(tensor_list_y) > 1:
        print(f"  井1形状: {tensor_list_y[1].shape}")

    return tensor_list_y, tensor_list_target, tensor_list_target_2


def prepare_batch_data(well_input, batch_size=2):
    """准备批次数据（用于ResNet和LSTM）"""
    n_samples = well_input.shape[0]
    n_batches = n_samples // batch_size

    # 只保留完整批次
    well_input_batched = well_input[:n_batches * batch_size]

    # 重塑为批次
    well_input_batched = well_input_batched.view(n_batches, batch_size,
                                                   well_input.shape[1],
                                                   well_input.shape[2])

    return well_input_batched, n_batches


def calculate_sample_mse(pred, target):
    """计算每个样本的MSE"""
    sample_mses = []
    n_samples = pred.shape[0]

    for i in range(n_samples):
        sample_pred = pred[i].flatten()
        sample_target = target[i].flatten()
        mse = np.mean((sample_pred - sample_target) ** 2)
        sample_mses.append(mse)

    return sample_mses


# ============================================================================
# 模型推理函数
# ============================================================================

def run_inference_on_model(model, model_name, tensor_list_y, tensor_list_target,
                           tensor_list_target_2, test_well_indices=[0, 1]):
    """
    在指定测试井上运行模型推理并计算样本级MSE

    Args:
        model: 训练好的模型
        model_name: 模型名称（用于输出）
        tensor_list_y: 输入数据列表
        tensor_list_target: 目标数据列表1
        tensor_list_target_2: 目标数据列表2
        test_well_indices: 要测试的井索引

    Returns:
        all_sample_mses: 所有样本的MSE列表
    """
    print(f"\n{'='*70}")
    print(f"运行 {model_name} 模型推理")
    print(f"{'='*70}")

    model.eval()
    all_sample_mses = []

    for well_idx in test_well_indices:
        print(f"\n处理井 {well_idx}...")

        # 获取该井的输入和目标数据
        well_input = tensor_list_y[well_idx].clone()
        well_target = torch.cat((tensor_list_target[well_idx][:,2:,:].clone(),
                                tensor_list_target_2[well_idx][:,1:3,:].clone()), dim=1)

        # 准备批次数据
        well_input_batched, n_batches = prepare_batch_data(well_input, batch_size=2)

        print(f"  样本数: {well_input.shape[0]}")
        print(f"  批次数: {n_batches}")
        print(f"  每批样本数: 2")

        # 推理
        well_preds = []
        with torch.no_grad():
            for batch_idx in range(n_batches):
                batch_input = well_input_batched[batch_idx].to(device)

                # 创建Data对象（模型期望接收有.x属性的对象）
                from types import SimpleNamespace
                data_obj = SimpleNamespace(x=batch_input)

                # 模型前向传播
                output = model(data_obj)

                # 移除CALI通道（索引0），只保留RSHA, RMED, RDEP, SP
                output = output[:, 1:, :]  # 从 (2, 5, 512) 变为 (2, 4, 512)

                well_preds.append(output.cpu())

        # 合并预测结果
        well_pred = torch.cat(well_preds, dim=0)
        well_target_aligned = well_target[:well_pred.shape[0]]

        print(f"  预测形状: {well_pred.shape}")
        print(f"  目标形状: {well_target_aligned.shape}")

        # 计算样本级MSE
        pred_np = well_pred.numpy()
        target_np = well_target_aligned.cpu().numpy()

        sample_mses = calculate_sample_mse(pred_np, target_np)
        all_sample_mses.extend(sample_mses)

        print(f"  井 {well_idx} 样本级MSE: 均值={np.mean(sample_mses):.6f}, "
              f"标准差={np.std(sample_mses):.6f}")

    print(f"\n{model_name} 总样本数: {len(all_sample_mses)}")
    print(f"{model_name} 总体MSE: 均值={np.mean(all_sample_mses):.6f}, "
          f"标准差={np.std(all_sample_mses):.6f}")

    return all_sample_mses


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("="*70)
    print("Baseline模型推理 - 用于R2-4统计显著性检验")
    print("="*70)

    # 模型路径
    model_dir = '/content/drive/MyDrive/logcompletion/model/ctra_exp_GCRR_GRSB/'
    resnet_path = model_dir + 'GCRR_GRSB200数据增强_ResNet.pt'
    lstm_path = model_dir + 'GCRR_GRSB200数据增强_LSTM.pt'

    # 加载测试数据
    tensor_list_y, tensor_list_target, tensor_list_target_2 = load_test_data()

    # 测试井索引（15/9-15和15/9-17）
    test_well_indices = [0, 1]

    # ========================================================================
    # 1. 加载并运行ResNet模型
    # ========================================================================
    print(f"\n{'='*70}")
    print("1. 加载ResNet模型")
    print(f"{'='*70}")

    # 定义ResNet模型结构
    num_blocks = [10, 10, 10, 10, 10, 4]
    resnet_model = ResNet(BasicBlock, num_blocks, input_channels=4, output_channels=5)
    resnet_model = resnet_model.to(device)

    # 加载权重（checkpoint格式）
    print(f"加载权重: {resnet_path}")
    checkpoint = torch.load(resnet_path, map_location=device)
    resnet_model.load_state_dict(checkpoint['model_state_dict'])
    print("ResNet模型加载成功！")

    # 运行推理
    resnet_sample_mses = run_inference_on_model(
        resnet_model, "ResNet",
        tensor_list_y, tensor_list_target, tensor_list_target_2,
        test_well_indices
    )

    # ========================================================================
    # 2. 加载并运行LSTM模型
    # ========================================================================
    print(f"\n{'='*70}")
    print("2. 加载Bi-LSTM模型")
    print(f"{'='*70}")

    # 定义LSTM模型结构
    lstm_model = LSTMModel(input_size=4, hidden_size=256, num_layers=3, output_size=5)
    lstm_model = lstm_model.to(device)

    # 加载权重（checkpoint格式）
    print(f"加载权重: {lstm_path}")
    checkpoint = torch.load(lstm_path, map_location=device)
    lstm_model.load_state_dict(checkpoint['model_state_dict'])
    print("Bi-LSTM模型加载成功！")

    # 运行推理
    lstm_sample_mses = run_inference_on_model(
        lstm_model, "Bi-LSTM",
        tensor_list_y, tensor_list_target, tensor_list_target_2,
        test_well_indices
    )

    # ========================================================================
    # 3. 输出格式化数据（用于填入fnpg_generation.py）
    # ========================================================================
    print(f"\n{'='*70}")
    print("3. 输出格式化数据")
    print(f"{'='*70}")

    print("\n" + "="*70)
    print("复制以下代码到 fnpg_generation.py 的 baseline_sample_mse 字典中：")
    print("="*70)
    print("\nbaseline_sample_mse = {")
    print(f"    'ResNet': np.array({resnet_sample_mses}),")
    print(f"    'Bi-LSTM': np.array({lstm_sample_mses}),")
    print("}")

    # 保存为numpy数组文件（方便后续使用）
    output_dir = '/content/drive/MyDrive/logcompletion/results/'
    os.makedirs(output_dir, exist_ok=True)

    np.save(output_dir + 'resnet_sample_mses.npy', np.array(resnet_sample_mses))
    np.save(output_dir + 'lstm_sample_mses.npy', np.array(lstm_sample_mses))

    print(f"\n数据已保存到:")
    print(f"  {output_dir}resnet_sample_mses.npy")
    print(f"  {output_dir}lstm_sample_mses.npy")

    # ========================================================================
    # 4. 统计摘要
    # ========================================================================
    print(f"\n{'='*70}")
    print("4. 统计摘要")
    print(f"{'='*70}")

    print(f"\nResNet:")
    print(f"  样本数: {len(resnet_sample_mses)}")
    print(f"  均值: {np.mean(resnet_sample_mses):.6f}")
    print(f"  标准差: {np.std(resnet_sample_mses):.6f}")
    print(f"  最小值: {np.min(resnet_sample_mses):.6f}")
    print(f"  最大值: {np.max(resnet_sample_mses):.6f}")

    print(f"\nBi-LSTM:")
    print(f"  样本数: {len(lstm_sample_mses)}")
    print(f"  均值: {np.mean(lstm_sample_mses):.6f}")
    print(f"  标准差: {np.std(lstm_sample_mses):.6f}")
    print(f"  最小值: {np.min(lstm_sample_mses):.6f}")
    print(f"  最大值: {np.max(lstm_sample_mses):.6f}")

    print(f"\n{'='*70}")
    print("完成！")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
