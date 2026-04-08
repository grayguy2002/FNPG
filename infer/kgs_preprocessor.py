"""
KGS Well Data Preprocessor
数据预处理：标准化、滑动窗口切分
遵循FNPG训练时的预处理流程
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DataNormalizer:
    """数据标准化类"""

    def __init__(self, ddof: int = 0, eps: float = 1e-8):
        self.stats = {}  # 存储每个特征的均值和标准差
        self.ddof = ddof
        self.eps = eps

    def fit(self, data: pd.DataFrame, features: List[str]):
        """
        计算标准化参数（均值和标准差）

        Args:
            data: 输入数据DataFrame
            features: 需要标准化的特征列表
        """
        print("\n正在计算标准化参数...")
        for feature in features:
            if feature in data.columns:
                valid_data = data[feature].dropna()
                self.stats[feature] = {
                    'mean': valid_data.mean(),
                    'std': float(valid_data.std(ddof=self.ddof))
                }
                if self.stats[feature]['std'] < self.eps:
                    self.stats[feature]['std'] = self.eps
                print(f"  {feature}: mean={self.stats[feature]['mean']:.4f}, "
                      f"std={self.stats[feature]['std']:.4f}")

    def fit_from_standard_data_csv(
        self,
        standard_data_csv: str,
        features: List[str],
        well_column: str = "WELL",
    ):
        """
        从FNPG训练时使用的 standard_data.csv 计算全局标准化参数（均值/标准差）。

        该逻辑对齐 paper/code/fnpg_generation.py::InverseTransformer.load_standard_params，
        但 std 默认使用 ddof=0（与 paper/code/standardardization_gcrr.py::mean_var_norm 一致）。
        """
        print("\n正在从训练集 standard_data.csv 计算全局标准化参数...")
        standard_data = pd.read_csv(standard_data_csv)

        for feature in features:
            if feature == well_column:
                continue
            if feature not in standard_data.columns:
                continue

            valid_data = standard_data[feature].dropna()
            self.stats[feature] = {
                'mean': valid_data.mean(),
                'std': float(valid_data.std(ddof=self.ddof)),
            }
            if self.stats[feature]['std'] < self.eps:
                self.stats[feature]['std'] = self.eps
            print(f"  {feature}: mean={self.stats[feature]['mean']:.4f}, "
                  f"std={self.stats[feature]['std']:.4f}")

    def transform(self, data: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        标准化数据

        Args:
            data: 输入数据DataFrame
            features: 需要标准化的特征列表

        Returns:
            标准化后的DataFrame
        """
        data_normalized = data.copy()

        for feature in features:
            if feature in self.stats and feature in data.columns:
                mean = self.stats[feature]['mean']
                std = self.stats[feature]['std']
                data_normalized[feature] = (data[feature] - mean) / std

        return data_normalized

    def inverse_transform(self, data: np.ndarray, feature: str) -> np.ndarray:
        """
        反标准化

        Args:
            data: 标准化后的数据
            feature: 特征名称

        Returns:
            反标准化后的数据
        """
        if feature not in self.stats:
            print(f"警告: 未找到 {feature} 的标准化参数")
            return data

        mean = self.stats[feature]['mean']
        std = self.stats[feature]['std']
        return data * std + mean

    def save_stats(self, filepath: str):
        """保存标准化参数"""
        stats_df = pd.DataFrame(self.stats).T
        stats_df.to_csv(filepath)
        print(f"\n标准化参数已保存到: {filepath}")

    def load_stats(self, filepath: str):
        """加载标准化参数"""
        stats_df = pd.read_csv(filepath, index_col=0)
        self.stats = stats_df.to_dict('index')
        print(f"\n已加载标准化参数: {filepath}")


class SlidingWindowProcessor:
    """滑动窗口处理器"""

    def __init__(self, window_size: int = 512, stride: int = 256):
        """
        初始化滑动窗口处理器

        Args:
            window_size: 窗口大小（FNPG默认512）
            stride: 滑动步长（默认为窗口大小的一半）
        """
        self.window_size = window_size
        self.stride = stride

    def create_windows(self,
                      data: pd.DataFrame,
                      input_features: List[str],
                      output_features: Optional[List[str]] = None
                      ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        创建滑动窗口

        Args:
            data: 输入数据DataFrame
            input_features: 输入特征列表
            output_features: 输出特征列表（可选，用于训练时）

        Returns:
            input_windows: 输入窗口 (N, num_input_features, window_size)
            output_windows: 输出窗口 (N, num_output_features, window_size) 或 None
            depth_indices: 每个窗口对应的深度索引
        """
        print(f"\n正在创建滑动窗口...")
        print(f"  窗口大小: {self.window_size}")
        print(f"  滑动步长: {self.stride}")

        # 获取数据
        input_data = data[input_features].values  # (n_samples, n_input_features)
        n_samples = len(input_data)

        # 计算窗口数量
        n_windows = (n_samples - self.window_size) // self.stride + 1
        print(f"  总样本数: {n_samples}")
        print(f"  窗口数量: {n_windows}")

        # 创建输入窗口
        input_windows = []
        output_windows = [] if output_features else None
        depth_indices = []

        for i in range(n_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size

            # 输入窗口 (window_size, n_input_features) -> (n_input_features, window_size)
            input_window = input_data[start_idx:end_idx, :].T
            input_windows.append(input_window)

            # 输出窗口（如果提供）
            if output_features:
                output_data = data[output_features].values
                output_window = output_data[start_idx:end_idx, :].T
                output_windows.append(output_window)

            # 记录深度索引（窗口中心点）
            center_idx = start_idx + self.window_size // 2
            depth_indices.append(center_idx)

        input_windows = np.array(input_windows)  # (N, n_input_features, window_size)

        if output_features:
            output_windows = np.array(output_windows)  # (N, n_output_features, window_size)

        depth_indices = np.array(depth_indices)

        print(f"  输入窗口形状: {input_windows.shape}")
        if output_windows is not None:
            print(f"  输出窗口形状: {output_windows.shape}")

        return input_windows, output_windows, depth_indices

    def reconstruct_from_windows(self,
                                windows: np.ndarray,
                                depth_indices: np.ndarray,
                                total_length: int,
                                method: str = 'average') -> np.ndarray:
        """
        从滑动窗口重建完整曲线

        Args:
            windows: 窗口数据 (N, n_features, window_size)
            depth_indices: 窗口中心点深度索引
            total_length: 重建后的总长度
            method: 重建方法 ('average', 'last', 'center')

        Returns:
            重建后的曲线 (total_length, n_features)
        """
        print(f"\n正在从窗口重建曲线 (方法: {method})...")

        n_windows, n_features, window_size = windows.shape

        # 初始化
        reconstructed = np.zeros((total_length, n_features))
        count = np.zeros(total_length)  # 记录每个点被覆盖的次数

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
                reconstructed[valid_start:valid_end, :] += windows[i, :, window_start:window_end].T
                count[valid_start:valid_end] += 1

            # 平均
            count[count == 0] = 1  # 避免除零
            reconstructed = reconstructed / count[:, np.newaxis]

        elif method == 'center':
            # 仅使用窗口中心点
            for i, center_idx in enumerate(depth_indices):
                if 0 <= center_idx < total_length:
                    window_center = window_size // 2
                    reconstructed[center_idx, :] = windows[i, :, window_center]
                    count[center_idx] += 1

        elif method == 'last':
            # 最后一个窗口覆盖
            for i, center_idx in enumerate(depth_indices):
                start_idx = center_idx - window_size // 2
                end_idx = start_idx + window_size

                valid_start = max(0, start_idx)
                valid_end = min(total_length, end_idx)

                window_start = valid_start - start_idx
                window_end = window_start + (valid_end - valid_start)

                reconstructed[valid_start:valid_end, :] = windows[i, :, window_start:window_end].T

        print(f"  重建形状: {reconstructed.shape}")
        return reconstructed


class KGSPreprocessor:
    """KGS数据完整预处理流程"""

    def __init__(
        self,
        window_size: int = 512,
        stride: int = 256,
        input_features: Optional[List[str]] = None,
        output_features: Optional[List[str]] = None,
    ):
        self.normalizer = DataNormalizer(ddof=0)
        self.window_processor = SlidingWindowProcessor(window_size, stride)
        # 注意：特征顺序必须与训练/推理模型期望的通道顺序完全一致
        self.input_features = input_features or ['GR', 'RHOB', 'NPHI', 'DTC']
        self.output_features = output_features or ['CALI', 'RSHA', 'RMED', 'RDEP', 'SP']

    def preprocess_for_inference(self,
                                 data: pd.DataFrame,
                                 save_stats: bool = True,
                                 stats_path: Optional[str] = None,
                                 standard_data_csv: Optional[str] = None,
                                 fill_input_nan_with_zero: bool = True,
                                 outlier_sigma_clip: Optional[float] = None,
                                 outlier_clip_features: Optional[List[str]] = None,
                                 ) -> Tuple[torch.Tensor, np.ndarray, pd.DataFrame]:
        """
        完整的推理预处理流程

        Args:
            data: 输入数据DataFrame（包含DEPT和所有特征）
            save_stats: 是否保存标准化参数
            stats_path: 标准化参数保存路径

        Returns:
            input_tensor: PyTorch tensor (N, 4, 512)
            depth_indices: 深度索引
            data_normalized: 标准化后的DataFrame
        """
        print("\n" + "="*60)
        print("开始KGS数据预处理流程")
        print("="*60)

        # 1. 计算标准化参数
        all_features = self.input_features + self.output_features
        if standard_data_csv:
            self.normalizer.fit_from_standard_data_csv(standard_data_csv, all_features)
            missing = [f for f in all_features if f not in self.normalizer.stats]
            if missing:
                raise ValueError(
                    "standard_data_csv 缺少以下特征列，无法对齐训练标准化流程: "
                    + ", ".join(missing)
                )
        else:
            self.normalizer.fit(data, all_features)

        # 1.5 参考训练数据分布进行异常值处理（近似对齐 standardardization_gcrr.py::handle_outlier）
        if outlier_sigma_clip is not None:
            clip_features = outlier_clip_features or self.input_features
            print(f"\n正在按 ±{outlier_sigma_clip}σ 裁剪异常值 (features={clip_features})...")
            data = data.copy()
            for feature in clip_features:
                if feature not in data.columns or feature not in self.normalizer.stats:
                    continue
                mean = self.normalizer.stats[feature]['mean']
                std = self.normalizer.stats[feature]['std']
                lower = mean - outlier_sigma_clip * std
                upper = mean + outlier_sigma_clip * std
                before_nan = data[feature].isna().sum()
                data.loc[(data[feature] < lower) | (data[feature] > upper), feature] = np.nan
                after_nan = data[feature].isna().sum()
                if after_nan > before_nan:
                    print(f"  {feature}: 新增 NaN {after_nan - before_nan} 个")

        if save_stats and stats_path:
            self.normalizer.save_stats(stats_path)

        # 2. 标准化数据
        print("\n正在标准化数据...")
        data_normalized = self.normalizer.transform(
            data,
            all_features
        )

        # 训练数据通常不包含 NaN；KGS推理若输入曲线缺失，使用 0（标准化后均值）填充，避免 NaN 传入模型
        if fill_input_nan_with_zero:
            data_normalized[self.input_features] = data_normalized[self.input_features].fillna(0.0)

        # 3. 创建滑动窗口
        input_windows, _, depth_indices = self.window_processor.create_windows(
            data_normalized,
            self.input_features,
            None  # 推理时不需要输出窗口
        )

        # 4. 转换为PyTorch tensor
        input_tensor = torch.from_numpy(input_windows).float()

        print("\n" + "="*60)
        print("预处理完成！")
        print("="*60)
        print(f"输入tensor形状: {input_tensor.shape}")
        print(f"深度索引数量: {len(depth_indices)}")

        return input_tensor, depth_indices, data_normalized


def main():
    """测试预处理器"""
    # 加载处理后的数据
    data_file = r"c:\Users\盖\Desktop\FNPG_JGRML\paper\KGS\1045063634_processed.csv"
    data = pd.read_csv(data_file)

    print(f"\n加载数据: {data_file}")
    print(f"数据形状: {data.shape}")
    print(f"深度范围: {data['DEPT'].min():.2f} - {data['DEPT'].max():.2f}")

    # 创建预处理器
    preprocessor = KGSPreprocessor(window_size=512, stride=256)

    # 预处理
    stats_path = r"c:\Users\盖\Desktop\FNPG_JGRML\paper\KGS\kgs_normalization_stats.csv"
    input_tensor, depth_indices, data_normalized = preprocessor.preprocess_for_inference(
        data,
        save_stats=True,
        stats_path=stats_path
    )

    # 保存预处理后的tensor
    tensor_path = r"c:\Users\盖\Desktop\FNPG_JGRML\paper\KGS\kgs_input_tensor.pt"
    torch.save({
        'input': input_tensor,
        'depth_indices': depth_indices,
        'depth_values': data['DEPT'].values[depth_indices]
    }, tensor_path)
    print(f"\n已保存输入tensor到: {tensor_path}")

    return input_tensor, depth_indices, preprocessor


if __name__ == "__main__":
    input_tensor, depth_indices, preprocessor = main()
