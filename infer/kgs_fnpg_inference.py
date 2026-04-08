"""
FNPG Inference Script for KGS Well
使用训练好的FNPG模型对KGS井进行推理预测
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# 添加代码路径以导入FNPG模型
# 自动查找 fnpg_train.py 所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # 上一级目录（code目录）

# 添加可能的代码路径
possible_paths = [
    parent_dir,  # infer的上一级目录
    r"c:\Users\盖\Desktop\FNPG_JGRML\paper\code",  # 本地路径
    "/content/drive/MyDrive/logcompletion/code",  # Colab路径
]

for path in possible_paths:
    if path not in sys.path and os.path.exists(path):
        sys.path.insert(0, path)

# 设置环境变量以阻止 fnpg_train.py 中的训练代码执行
# 这样导入时只会加载类定义，不会执行数据加载和训练
os.environ['FNPG_INFERENCE_MODE'] = '1'

# 导入FNPG模型（从fnpg_train.py）
# 注意：需要确保fnpg_train.py中的所有依赖类都被正确导入
try:
    from fnpg_train import FNPG, Data
    print("✓ 成功导入FNPG模型")
except ImportError as e:
    print(f"✗ 导入FNPG模型失败: {e}")
    print(f"请确保fnpg_train.py在以下路径之一:")
    for path in possible_paths:
        print(f"  - {path}")
    print(f"\n当前Python路径: {sys.path[:3]}")
    sys.exit(1)
except Exception as e:
    print(f"✗ 导入时发生错误: {e}")
    print(f"这可能是因为 fnpg_train.py 在导入时执行了训练代码")
    print(f"请检查 fnpg_train.py 是否将训练代码放在了 if __name__ == '__main__': 块中")
    import traceback
    traceback.print_exc()
    sys.exit(1)


class FNPGInference:
    """FNPG推理类"""

    def __init__(self,
                 model_path: str,
                 device: Optional[str] = None):
        """
        初始化FNPG推理器

        Args:
            model_path: 训练好的模型路径（.pt文件）
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.model_path = model_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

        print(f"\n初始化FNPG推理器...")
        print(f"  模型路径: {model_path}")
        print(f"  计算设备: {self.device}")

    def load_model(self,
                   graph_channels: list = [512, 256, 128, 64, 32],
                   batch_size: int = 2,
                   fea_litho: list = ['GR', 'RHOB', 'NPHI', 'DTC'],
                   head: int = 1,
                   drop: float = 0,
                   modes1: int = 32):
        """
        加载训练好的模型

        Args:
            graph_channels: 图通道数列表
            batch_size: 批次大小
            fea_litho: 输入特征列表
            head: 注意力头数
            drop: Dropout比例
            modes1: 傅里叶模式数（kmax）
        """
        print(f"\n正在加载FNPG模型...")

        # 创建模型实例
        self.model = FNPG(
            gnn_ch=graph_channels,
            batchsize=batch_size,
            fea_litho=fea_litho,
            head=head,
            drop=drop,
            modes1=modes1
        ).to(self.device)

        # 加载模型权重
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # 检查checkpoint格式
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"  ✓ 加载检查点: epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"  ✓ 加载模型权重")

            # 设置为评估模式
            self.model.eval()
            print(f"  ✓ 模型已设置为评估模式")
        else:
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        return self.model

    def predict(self,
                input_tensor: torch.Tensor,
                batch_size: int = 32) -> np.ndarray:
        """
        对输入数据进行预测

        Args:
            input_tensor: 输入数据 (N, 4, 512)
            batch_size: 推理批次大小

        Returns:
            预测结果 (N, 5, 512)
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用load_model()")

        print(f"\n正在进行FNPG推理...")
        print(f"  输入形状: {input_tensor.shape}")
        print(f"  批次大小: {batch_size}")

        # 确保输入在正确的设备上
        input_tensor = input_tensor.to(self.device)

        # 分批推理
        n_samples = input_tensor.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        predictions = []

        with torch.no_grad():
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)

                # 获取批次数据
                batch_input = input_tensor[start_idx:end_idx]

                # 创建Data对象
                batch_data = Data(input_sam=batch_input, label=None)

                # 推理
                batch_output = self.model(batch_data)  # (batch, 5, 512)

                predictions.append(batch_output.cpu().numpy())

                if (i + 1) % 10 == 0:
                    print(f"  进度: {i+1}/{n_batches} 批次")

        # 合并所有批次的预测
        predictions = np.concatenate(predictions, axis=0)

        print(f"  ✓ 推理完成！")
        print(f"  输出形状: {predictions.shape}")

        return predictions


class ResultReconstructor:
    """从窗口预测结果重建完整曲线"""

    def __init__(self, window_size: int = 512, stride: int = 256):
        self.window_size = window_size
        self.stride = stride

    def reconstruct(self,
                   predictions: np.ndarray,
                   depth_indices: np.ndarray,
                   total_length: int,
                   method: str = 'average') -> np.ndarray:
        """
        从窗口预测重建完整曲线

        Args:
            predictions: 窗口预测结果 (N, n_features, window_size)
            depth_indices: 窗口中心点深度索引
            total_length: 重建后的总长度
            method: 重建方法 ('average', 'center')

        Returns:
            重建后的曲线 (total_length, n_features)
        """
        print(f"\n正在重建完整曲线 (方法: {method})...")

        n_windows, n_features, window_size = predictions.shape

        # 初始化
        reconstructed = np.zeros((total_length, n_features))
        count = np.zeros(total_length)

        if method == 'average':
            # 平均重叠区域
            for i, center_idx in enumerate(depth_indices):
                start_idx = center_idx - window_size // 2
                end_idx = start_idx + window_size

                # 确保索引在有效范围内
                valid_start = max(0, start_idx)
                valid_end = min(total_length, end_idx)

                window_start = valid_start - start_idx
                window_end = window_start + (valid_end - valid_start)

                # 累加窗口数据
                reconstructed[valid_start:valid_end, :] += predictions[i, :, window_start:window_end].T
                count[valid_start:valid_end] += 1

            # 平均
            count[count == 0] = 1
            reconstructed = reconstructed / count[:, np.newaxis]

        elif method == 'center':
            # 仅使用窗口中心点
            for i, center_idx in enumerate(depth_indices):
                if 0 <= center_idx < total_length:
                    window_center = window_size // 2
                    reconstructed[center_idx, :] = predictions[i, :, window_center]

        print(f"  ✓ 重建完成！形状: {reconstructed.shape}")

        return reconstructed


def main():
    """主函数：完整的推理流程"""

    # ========== 配置 ==========
    MODEL_PATH = r"c:\Users\盖\Desktop\FNPG_JGRML\paper\code\models\FNPG_kmax32_best.pt"
    INPUT_TENSOR_PATH = r"c:\Users\盖\Desktop\FNPG_JGRML\paper\KGS\kgs_input_tensor.pt"
    OUTPUT_DIR = r"c:\Users\盖\Desktop\FNPG_JGRML\paper\KGS"

    # 模型参数（需要与训练时一致）
    MODEL_CONFIG = {
        'graph_channels': [512, 256, 128, 64, 32],
        'batch_size': 2,
        'fea_litho': ['GR', 'RHOB', 'NPHI', 'DTC'],
        'head': 1,
        'drop': 0,
        'modes1': 32  # kmax参数
    }

    OUTPUT_FEATURES = ['CALI', 'RSHA', 'RMED', 'RDEP', 'SP']

    print("="*60)
    print("FNPG KGS井推理流程")
    print("="*60)

    # ========== 1. 加载输入数据 ==========
    print(f"\n正在加载预处理数据: {INPUT_TENSOR_PATH}")
    data = torch.load(INPUT_TENSOR_PATH)
    input_tensor = data['input']
    depth_indices = data['depth_indices']
    depth_values = data['depth_values']

    print(f"  输入tensor形状: {input_tensor.shape}")
    print(f"  深度索引数量: {len(depth_indices)}")
    print(f"  深度范围: {depth_values.min():.2f} - {depth_values.max():.2f}")

    # ========== 2. 初始化并加载模型 ==========
    inferencer = FNPGInference(MODEL_PATH)
    inferencer.load_model(**MODEL_CONFIG)

    # ========== 3. 推理 ==========
    predictions = inferencer.predict(input_tensor, batch_size=32)

    # ========== 4. 重建完整曲线 ==========
    # 计算原始数据长度
    max_depth_idx = depth_indices.max() + 256  # 考虑最后一个窗口
    reconstructor = ResultReconstructor(window_size=512, stride=256)
    reconstructed = reconstructor.reconstruct(
        predictions,
        depth_indices,
        total_length=max_depth_idx,
        method='average'
    )

    # ========== 5. 保存结果 ==========
    # 保存为numpy数组
    np_output_path = os.path.join(OUTPUT_DIR, "kgs_predictions_normalized.npy")
    np.save(np_output_path, reconstructed)
    print(f"\n✓ 已保存标准化预测结果: {np_output_path}")

    # 保存为CSV（标准化数据）
    csv_output_path = os.path.join(OUTPUT_DIR, "kgs_predictions_normalized.csv")
    df_predictions = pd.DataFrame(
        reconstructed,
        columns=OUTPUT_FEATURES
    )
    df_predictions.to_csv(csv_output_path, index=False)
    print(f"✓ 已保存CSV文件: {csv_output_path}")

    print("\n" + "="*60)
    print("推理完成！")
    print("="*60)
    print(f"预测曲线: {OUTPUT_FEATURES}")
    print(f"数据点数: {len(reconstructed)}")

    return reconstructed, OUTPUT_FEATURES


if __name__ == "__main__":
    predictions, output_features = main()
