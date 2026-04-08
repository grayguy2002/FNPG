"""
Inverse Transformation and Visualization
反标准化并可视化KGS井的推理结果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')


class InverseNormalizer:
    """反标准化工具"""

    def __init__(self, stats_path: str):
        """
        初始化反标准化器

        Args:
            stats_path: 标准化参数文件路径（CSV格式）
        """
        self.stats_path = stats_path
        self.stats = self.load_stats()

    def load_stats(self) -> dict:
        """加载标准化参数"""
        print(f"\n正在加载标准化参数: {self.stats_path}")
        stats_df = pd.read_csv(self.stats_path, index_col=0)

        stats_dict = {}
        for feature in stats_df.index:
            stats_dict[feature] = {
                'mean': stats_df.loc[feature, 'mean'],
                'std': stats_df.loc[feature, 'std']
            }
            print(f"  {feature}: mean={stats_dict[feature]['mean']:.4f}, "
                  f"std={stats_dict[feature]['std']:.4f}")

        return stats_dict

    def inverse_transform(self,
                         data: np.ndarray,
                         features: List[str]) -> np.ndarray:
        """
        反标准化

        Args:
            data: 标准化后的数据 (n_samples, n_features)
            features: 特征名称列表

        Returns:
            反标准化后的数据
        """
        print(f"\n正在反标准化数据...")
        data_denorm = data.copy()

        for i, feature in enumerate(features):
            if feature in self.stats:
                mean = self.stats[feature]['mean']
                std = self.stats[feature]['std']
                data_denorm[:, i] = data[:, i] * std + mean
                print(f"  ✓ {feature}: 反标准化完成")
            else:
                print(f"  ✗ 警告: 未找到 {feature} 的标准化参数")

        return data_denorm


class ResultVisualizer:
    """结果可视化工具"""

    def __init__(self, figsize=(20, 12)):
        self.figsize = figsize

    def plot_comparison(self,
                       depth: np.ndarray,
                       original_data: pd.DataFrame,
                       predicted_data: pd.DataFrame,
                       features: List[str],
                       robust_xlim: bool = True,
                       xlim_percentiles: tuple = (1, 99),
                       use_log_for_resistivity: bool = True,
                       output_path: Optional[str] = None):
        """
        绘制原始数据与预测数据的对比图

        Args:
            depth: 深度值
            original_data: 原始数据DataFrame
            predicted_data: 预测数据DataFrame
            features: 要绘制的特征列表
            output_path: 保存路径（可选）
        """
        print(f"\n正在绘制对比图...")

        n_features = len(features)
        fig, axes = plt.subplots(1, n_features, figsize=self.figsize, sharey=True)

        if n_features == 1:
            axes = [axes]

        for i, feature in enumerate(features):
            ax = axes[i]

            # 绘制原始数据
            if feature in original_data.columns:
                original = original_data[feature].values
                if use_log_for_resistivity and feature in {'RSHA', 'RMED', 'RDEP'}:
                    original = np.where(original > 0, original, np.nan)
                ax.plot(original, depth, 'b-', label='Original', linewidth=1.5, alpha=0.7)

            # 绘制预测数据
            if feature in predicted_data.columns:
                predicted = predicted_data[feature].values
                # 确保预测数据长度匹配
                if len(predicted) < len(depth):
                    # 补齐预测数据
                    predicted = np.pad(predicted, (0, len(depth) - len(predicted)),
                                     mode='edge')
                predicted = predicted[:len(depth)]
                if use_log_for_resistivity and feature in {'RSHA', 'RMED', 'RDEP'}:
                    predicted = np.where(predicted > 0, predicted, np.nan)
                ax.plot(predicted, depth, 'r--',
                       label='Predicted', linewidth=1.5, alpha=0.7)

            # 电阻率建议对数坐标，避免“蓝线很窄”（线性坐标下高阻段会把轴拉得很大）
            if use_log_for_resistivity and feature in {'RSHA', 'RMED', 'RDEP'}:
                ax.set_xscale('log')
                ax.grid(True, which='both', alpha=0.25)
            else:
                ax.grid(True, alpha=0.3)

            # 鲁棒缩放：用分位数设置xlim，避免少量极端值把轴拉爆
            if robust_xlim:
                series = []
                if feature in original_data.columns:
                    series.append(original_data[feature].values)
                if feature in predicted_data.columns:
                    series.append(predicted_data[feature].values)
                if series:
                    vals = np.concatenate(series, axis=0)
                    vals = vals[np.isfinite(vals)]
                    if use_log_for_resistivity and feature in {'RSHA', 'RMED', 'RDEP'}:
                        vals = vals[vals > 0]
                    if vals.size > 10:
                        lo, hi = np.nanpercentile(vals, list(xlim_percentiles))
                        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                            ax.set_xlim(lo, hi)

            ax.set_xlabel(feature, fontsize=12, fontweight='bold')
            ax.set_title(f'{feature}', fontsize=14, fontweight='bold')
            ax.legend(loc='best')

            # 反转Y轴（深度增加向下）
            ax.invert_yaxis()

        # 设置Y轴标签（只在第一个子图）
        axes[0].set_ylabel('Depth (ft)', fontsize=12, fontweight='bold')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ 对比图已保存: {output_path}")

        plt.show()

    def plot_predictions_only(self,
                             depth: np.ndarray,
                             predicted_data: pd.DataFrame,
                             features: List[str],
                             robust_xlim: bool = True,
                             xlim_percentiles: tuple = (1, 99),
                             use_log_for_resistivity: bool = True,
                             output_path: Optional[str] = None):
        """
        仅绘制预测数据

        Args:
            depth: 深度值
            predicted_data: 预测数据DataFrame
            features: 要绘制的特征列表
            output_path: 保存路径（可选）
        """
        print(f"\n正在绘制预测曲线...")

        n_features = len(features)
        fig, axes = plt.subplots(1, n_features, figsize=self.figsize, sharey=True)

        if n_features == 1:
            axes = [axes]

        for i, feature in enumerate(features):
            ax = axes[i]

            if feature in predicted_data.columns:
                predicted = predicted_data[feature].values
                # 确保预测数据长度匹配
                if len(predicted) < len(depth):
                    predicted = np.pad(predicted, (0, len(depth) - len(predicted)),
                                     mode='edge')
                predicted = predicted[:len(depth)]
                if use_log_for_resistivity and feature in {'RSHA', 'RMED', 'RDEP'}:
                    predicted = np.where(predicted > 0, predicted, np.nan)
                ax.plot(predicted, depth, 'r-', linewidth=2)

            ax.set_xlabel(feature, fontsize=12, fontweight='bold')
            ax.set_title(f'{feature} (FNPG Predicted)', fontsize=14, fontweight='bold')
            if use_log_for_resistivity and feature in {'RSHA', 'RMED', 'RDEP'}:
                ax.set_xscale('log')
                ax.grid(True, which='both', alpha=0.25)
            else:
                ax.grid(True, alpha=0.3)

            if robust_xlim and feature in predicted_data.columns:
                vals = predicted_data[feature].values
                vals = vals[np.isfinite(vals)]
                if use_log_for_resistivity and feature in {'RSHA', 'RMED', 'RDEP'}:
                    vals = vals[vals > 0]
                if vals.size > 10:
                    lo, hi = np.nanpercentile(vals, list(xlim_percentiles))
                    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                        ax.set_xlim(lo, hi)

            # 反转Y轴
            ax.invert_yaxis()

        axes[0].set_ylabel('Depth (ft)', fontsize=12, fontweight='bold')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ 预测图已保存: {output_path}")

        plt.show()

    def plot_statistics(self,
                       original_data: pd.DataFrame,
                       predicted_data: pd.DataFrame,
                       features: List[str],
                       output_path: Optional[str] = None):
        """
        绘制统计对比图（箱线图）

        Args:
            original_data: 原始数据DataFrame
            predicted_data: 预测数据DataFrame
            features: 特征列表
            output_path: 保存路径
        """
        print(f"\n正在绘制统计对比图...")

        fig, axes = plt.subplots(2, (len(features) + 1) // 2,
                                figsize=(15, 8))
        axes = axes.flatten()

        for i, feature in enumerate(features):
            ax = axes[i]

            data_to_plot = []
            labels = []

            if feature in original_data.columns:
                data_to_plot.append(original_data[feature].dropna())
                labels.append('Original')

            if feature in predicted_data.columns:
                data_to_plot.append(predicted_data[feature].dropna())
                labels.append('Predicted')

            ax.boxplot(data_to_plot, labels=labels)
            ax.set_title(feature, fontweight='bold')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ 统计图已保存: {output_path}")

        plt.show()


def calculate_metrics(original: np.ndarray,
                     predicted: np.ndarray,
                     feature_name: str) -> dict:
    """
    计算评估指标

    Args:
        original: 原始数据
        predicted: 预测数据
        feature_name: 特征名称

    Returns:
        评估指标字典
    """
    # 移除NaN值
    mask = ~(np.isnan(original) | np.isnan(predicted))
    original = original[mask]
    predicted = predicted[mask]

    if len(original) == 0:
        return {
            'feature': feature_name,
            'mae': np.nan,
            'rmse': np.nan,
            'r2': np.nan,
            'correlation': np.nan
        }

    # MAE
    mae = np.mean(np.abs(original - predicted))

    # RMSE
    rmse = np.sqrt(np.mean((original - predicted) ** 2))

    # R²
    ss_res = np.sum((original - predicted) ** 2)
    ss_tot = np.sum((original - np.mean(original)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

    # 相关系数
    correlation = np.corrcoef(original, predicted)[0, 1]

    return {
        'feature': feature_name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'correlation': correlation
    }


def main():
    """主函数：反标准化和可视化"""

    # ========== 配置 ==========
    KGS_DIR = r"c:\Users\盖\Desktop\FNPG_JGRML\paper\KGS"
    STATS_PATH = f"{KGS_DIR}/kgs_normalization_stats.csv"
    NORMALIZED_PRED_PATH = f"{KGS_DIR}/kgs_predictions_normalized.csv"
    PROCESSED_DATA_PATH = f"{KGS_DIR}/1045063634_processed.csv"

    OUTPUT_FEATURES = ['CALI', 'RSHA', 'RMED', 'RDEP', 'SP']

    print("="*60)
    print("KGS井推理结果后处理")
    print("="*60)

    # ========== 1. 加载标准化预测结果 ==========
    print(f"\n正在加载标准化预测结果...")
    pred_normalized = pd.read_csv(NORMALIZED_PRED_PATH)
    print(f"  预测数据形状: {pred_normalized.shape}")

    # ========== 2. 反标准化 ==========
    inverse_normalizer = InverseNormalizer(STATS_PATH)
    pred_denormalized = inverse_normalizer.inverse_transform(
        pred_normalized.values,
        OUTPUT_FEATURES
    )

    # 转换为DataFrame
    df_pred_denorm = pd.DataFrame(pred_denormalized, columns=OUTPUT_FEATURES)

    # 保存反标准化结果
    denorm_output_path = f"{KGS_DIR}/kgs_predictions_denormalized.csv"
    df_pred_denorm.to_csv(denorm_output_path, index=False)
    print(f"\n✓ 已保存反标准化预测结果: {denorm_output_path}")

    # ========== 3. 加载原始数据（用于对比） ==========
    print(f"\n正在加载原始处理数据...")
    df_original = pd.read_csv(PROCESSED_DATA_PATH)
    depth = df_original['DEPT'].values

    # 对齐长度
    min_len = min(len(depth), len(df_pred_denorm))
    depth = depth[:min_len]
    df_original = df_original.iloc[:min_len]
    df_pred_denorm = df_pred_denorm.iloc[:min_len]

    # ========== 4. 计算评估指标 ==========
    print("\n" + "="*60)
    print("评估指标（与原始数据对比）")
    print("="*60)

    metrics_list = []
    for feature in OUTPUT_FEATURES:
        if feature in df_original.columns:
            metrics = calculate_metrics(
                df_original[feature].values,
                df_pred_denorm[feature].values,
                feature
            )
            metrics_list.append(metrics)

            print(f"\n{feature}:")
            print(f"  MAE:         {metrics['mae']:.4f}")
            print(f"  RMSE:        {metrics['rmse']:.4f}")
            print(f"  R²:          {metrics['r2']:.4f}")
            print(f"  Correlation: {metrics['correlation']:.4f}")

    # 保存指标
    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = f"{KGS_DIR}/kgs_evaluation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n✓ 已保存评估指标: {metrics_path}")

    # ========== 5. 可视化 ==========
    visualizer = ResultVisualizer(figsize=(20, 12))

    # 绘制对比图
    comparison_plot_path = f"{KGS_DIR}/kgs_comparison_plot.png"
    visualizer.plot_comparison(
        depth,
        df_original,
        df_pred_denorm,
        OUTPUT_FEATURES,
        output_path=comparison_plot_path
    )

    # 绘制仅预测曲线
    prediction_plot_path = f"{KGS_DIR}/kgs_prediction_plot.png"
    visualizer.plot_predictions_only(
        depth,
        df_pred_denorm,
        OUTPUT_FEATURES,
        output_path=prediction_plot_path
    )

    # 绘制统计对比
    stats_plot_path = f"{KGS_DIR}/kgs_statistics_plot.png"
    visualizer.plot_statistics(
        df_original,
        df_pred_denorm,
        OUTPUT_FEATURES,
        output_path=stats_plot_path
    )

    print("\n" + "="*60)
    print("后处理完成！")
    print("="*60)

    return df_pred_denorm, metrics_df


if __name__ == "__main__":
    predictions, metrics = main()
