"""
KGS Well Complete Inference Pipeline
KGS井完整推理流程 - 一键运行所有步骤

使用方法:
    python kgs_pipeline.py --las_file <path_to_las> --model_path <path_to_model>

或使用默认配置:
    python kgs_pipeline.py
"""

import argparse
import os
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kgs_las_reader import KGSLASReader
from kgs_preprocessor import KGSPreprocessor, DataNormalizer
from kgs_fnpg_inference import FNPGInference, ResultReconstructor
from kgs_postprocess import InverseNormalizer, ResultVisualizer, calculate_metrics

import torch
import pandas as pd
import numpy as np


class KGSInferencePipeline:
    """KGS井完整推理流程"""

    def __init__(self, config: dict):
        self.config = config
        self.results = {}

    def step1_read_las(self):
        """步骤1: 读取LAS文件"""
        print("\n" + "="*60)
        print("步骤 1/5: 读取LAS文件")
        print("="*60)

        reader = KGSLASReader(self.config['las_file'])
        df = reader.read_las()

        # 提取曲线
        curve_mapping = {
            'GR': 'GR',
            'NPHI': 'NPHI',
            'RHOB': 'RHOB',
            'DTC': 'DELTAT',
            'CALI': 'CALI',
            'RSHA': 'RT30',
            'RMED': 'RT60',
            'RDEP': 'RT90',
            'SP': 'SP'
        }

        df_extracted = reader.extract_curves(curve_mapping)
        df_clean = reader.clean_data(df_extracted)

        # 移除全空值段
        input_curves = ['GR', 'NPHI', 'RHOB', 'DTC']
        df_valid = reader.remove_null_sections(df_clean, input_curves)

        # 插值填充
        all_curves = list(curve_mapping.keys())
        df_final = reader.interpolate_missing(df_valid, all_curves, limit=10)

        # 保存
        processed_path = os.path.join(self.config['output_dir'],
                                     "kgs_processed.csv")
        df_final.to_csv(processed_path, index=False)
        print(f"\n✓ 步骤1完成！已保存处理后数据: {processed_path}")

        # 可选：按深度范围裁剪（用于局部切片推理）
        depth_min = self.config.get('depth_min')
        depth_max = self.config.get('depth_max')
        if depth_min is not None or depth_max is not None:
            before_len = len(df_final)
            mask = np.ones(before_len, dtype=bool)
            if depth_min is not None:
                mask &= df_final['DEPT'].values >= depth_min
            if depth_max is not None:
                mask &= df_final['DEPT'].values <= depth_max
            df_final = df_final.loc[mask].reset_index(drop=True)
            print(f"\n已按深度范围裁剪数据: {depth_min} - {depth_max} ft")
            print(f"  裁剪前点数: {before_len}")
            print(f"  裁剪后点数: {len(df_final)}")
            if len(df_final) == 0:
                raise ValueError("深度裁剪后为空，请检查 --depth_min/--depth_max 与 LAS 深度单位是否一致")

            processed_slice_path = os.path.join(self.config['output_dir'], "kgs_processed_slice.csv")
            df_final.to_csv(processed_slice_path, index=False)
            print(f"  已保存裁剪后数据: {processed_slice_path}")

        self.results['processed_data'] = df_final
        return df_final

    def step2_preprocess(self):
        """步骤2: 数据预处理和滑动窗口"""
        print("\n" + "="*60)
        print("步骤 2/5: 数据预处理")
        print("="*60)

        preprocessor = KGSPreprocessor(
            window_size=self.config['window_size'],
            stride=self.config['stride'],
            input_features=self.config['input_features'],
            output_features=self.config['output_features'],
        )
        print(f"\n预处理输入特征顺序: {preprocessor.input_features}")
        print(f"模型期望输入特征顺序: {self.config['input_features']}")

        stats_path = os.path.join(self.config['output_dir'],
                                 "kgs_normalization_stats.csv")

        input_tensor, depth_indices, data_normalized = preprocessor.preprocess_for_inference(
            self.results['processed_data'],
            save_stats=True,
            stats_path=stats_path,
            standard_data_csv=self.config.get('standard_data_csv'),
            fill_input_nan_with_zero=True,
            outlier_sigma_clip=self.config.get('outlier_sigma_clip'),
        )

        # 保存tensor
        tensor_path = os.path.join(self.config['output_dir'],
                                  "kgs_input_tensor.pt")
        torch.save({
            'input': input_tensor,
            'depth_indices': depth_indices,
            'depth_values': self.results['processed_data']['DEPT'].values[depth_indices]
        }, tensor_path)

        print(f"\n✓ 步骤2完成！已保存输入tensor: {tensor_path}")

        self.results['input_tensor'] = input_tensor
        self.results['depth_indices'] = depth_indices
        self.results['normalizer'] = preprocessor.normalizer
        self.results['processed_data_normalized'] = data_normalized

        return input_tensor, depth_indices

    def step3_inference(self):
        """步骤3: FNPG推理"""
        print("\n" + "="*60)
        print("步骤 3/5: FNPG模型推理")
        print("="*60)

        # 初始化推理器
        inferencer = FNPGInference(self.config['model_path'])

        # 加载模型
        inferencer.load_model(
            graph_channels=self.config['graph_channels'],
            batch_size=self.config['batch_size'],
            fea_litho=self.config['input_features'],
            head=self.config['head'],
            drop=self.config['drop'],
            modes1=self.config['kmax']
        )

        # 推理
        predictions = inferencer.predict(
            self.results['input_tensor'],
            batch_size=self.config['inference_batch_size']
        )

        # 重建完整曲线
        max_depth_idx = self.results['depth_indices'].max() + self.config['stride']
        reconstructor = ResultReconstructor(
            window_size=self.config['window_size'],
            stride=self.config['stride']
        )

        reconstructed = reconstructor.reconstruct(
            predictions,
            self.results['depth_indices'],
            total_length=max_depth_idx,
            method='average'
        )

        # 保存标准化预测结果
        pred_norm_path = os.path.join(self.config['output_dir'],
                                     "kgs_predictions_normalized.csv")
        df_pred_norm = pd.DataFrame(reconstructed, columns=self.config['output_features'])
        df_pred_norm.to_csv(pred_norm_path, index=False)

        print(f"\n✓ 步骤3完成！已保存标准化预测: {pred_norm_path}")

        self.results['predictions_normalized'] = df_pred_norm
        return df_pred_norm

    def step4_denormalize(self):
        """步骤4: 反标准化"""
        print("\n" + "="*60)
        print("步骤 4/5: 反标准化")
        print("="*60)

        stats_path = os.path.join(self.config['output_dir'],
                                 "kgs_normalization_stats.csv")

        inverse_normalizer = InverseNormalizer(stats_path)

        pred_denormalized = inverse_normalizer.inverse_transform(
            self.results['predictions_normalized'].values,
            self.config['output_features']
        )

        df_pred_denorm = pd.DataFrame(pred_denormalized,
                                     columns=self.config['output_features'])

        # 保存
        denorm_path = os.path.join(self.config['output_dir'],
                                  "kgs_predictions_denormalized.csv")
        df_pred_denorm.to_csv(denorm_path, index=False)

        print(f"\n✓ 步骤4完成！已保存反标准化预测: {denorm_path}")

        self.results['predictions_denormalized'] = df_pred_denorm
        return df_pred_denorm

    def step5_visualize(self):
        """步骤5: 可视化和评估"""
        print("\n" + "="*60)
        print("步骤 5/5: 可视化和评估")
        print("="*60)

        # 加载原始数据
        df_original = self.results['processed_data']
        df_predicted = self.results['predictions_denormalized']
        df_original_normalized = self.results.get('processed_data_normalized')
        df_predicted_normalized = self.results.get('predictions_normalized')

        # 对齐长度
        min_len = min(len(df_original), len(df_predicted))
        df_original = df_original.iloc[:min_len]
        df_predicted = df_predicted.iloc[:min_len]
        depth = df_original['DEPT'].values

        # 计算评估指标
        print("\n评估指标:")
        print("-" * 60)

        metrics_list = []
        for feature in self.config['output_features']:
            if feature in df_original.columns:
                metrics = calculate_metrics(
                    df_original[feature].values,
                    df_predicted[feature].values,
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
        metrics_path = os.path.join(self.config['output_dir'],
                                   "kgs_evaluation_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)

        # 可视化
        visualizer = ResultVisualizer(figsize=(20, 12))

        # 对比图
        comparison_path = os.path.join(self.config['output_dir'],
                                      "kgs_comparison_plot.png")
        visualizer.plot_comparison(
            depth, df_original, df_predicted,
            self.config['output_features'],
            robust_xlim=True,
            xlim_percentiles=(1, 99),
            use_log_for_resistivity=True,
            output_path=comparison_path
        )

        # 标准化空间对比图（原始输出曲线 vs 预测输出曲线，均为标准化后）
        if df_original_normalized is not None and df_predicted_normalized is not None:
            min_len_norm = min(len(df_original_normalized), len(df_predicted_normalized))
            df_orig_norm = df_original_normalized.iloc[:min_len_norm]
            df_pred_norm = df_predicted_normalized.iloc[:min_len_norm]
            depth_norm = df_orig_norm['DEPT'].values if 'DEPT' in df_orig_norm.columns else depth[:min_len_norm]

            comparison_norm_path = os.path.join(
                self.config['output_dir'],
                "kgs_comparison_plot_normalized.png",
            )
            visualizer.plot_comparison(
                depth_norm, df_orig_norm, df_pred_norm,
                self.config['output_features'],
                robust_xlim=True,
                xlim_percentiles=(1, 99),
                use_log_for_resistivity=False,  # 标准化后可能有负值，不能用log轴
                output_path=comparison_norm_path,
            )

        # 预测图
        prediction_path = os.path.join(self.config['output_dir'],
                                      "kgs_prediction_plot.png")
        visualizer.plot_predictions_only(
            depth, df_predicted,
            self.config['output_features'],
            robust_xlim=True,
            xlim_percentiles=(1, 99),
            use_log_for_resistivity=True,
            output_path=prediction_path
        )

        print(f"\n✓ 步骤5完成！")
        print(f"  评估指标: {metrics_path}")
        print(f"  对比图: {comparison_path}")
        print(f"  预测图: {prediction_path}")

        self.results['metrics'] = metrics_df
        return metrics_df

    def run(self):
        """运行完整流程"""
        print("\n" + "="*60)
        print("KGS井FNPG推理完整流程")
        print("="*60)
        print(f"\nLAS文件: {self.config['las_file']}")
        print(f"模型文件: {self.config['model_path']}")
        print(f"输出目录: {self.config['output_dir']}")
        if self.config.get('standard_data_csv'):
            print(f"标准化参数: 使用训练集 {self.config['standard_data_csv']}")
        else:
            print("标准化参数: 使用当前井数据统计（可能与训练不一致）")
        if self.config.get('outlier_sigma_clip') is not None:
            print(f"异常值处理: ±{self.config['outlier_sigma_clip']}σ 裁剪")
        if self.config.get('depth_min') is not None or self.config.get('depth_max') is not None:
            print(f"推理深度切片: {self.config.get('depth_min')} - {self.config.get('depth_max')} ft")

        try:
            self.step1_read_las()
            self.step2_preprocess()
            self.step3_inference()
            self.step4_denormalize()
            self.step5_visualize()

            print("\n" + "="*60)
            print("✓ 所有步骤完成！")
            print("="*60)

            return self.results

        except Exception as e:
            print(f"\n✗ 错误: {e}")
            import traceback
            traceback.print_exc()
            return None


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="KGS井FNPG推理流程")

    parser.add_argument('--las_file', type=str,
                       default=r"c:\Users\盖\Desktop\FNPG_JGRML\paper\KGS\1045063634.las",
                       help='LAS文件路径')

    parser.add_argument('--model_path', type=str,
                       default=r"c:\Users\盖\Desktop\FNPG_JGRML\paper\code\models\FNPG_kmax32_best.pt",
                       help='FNPG模型路径')

    parser.add_argument('--output_dir', type=str,
                       default=r"c:\Users\盖\Desktop\FNPG_JGRML\paper\KGS",
                       help='输出目录')

    parser.add_argument('--standard_data_csv', type=str, default=None,
                       help='FNPG训练集 standard_data.csv 路径（用于全局标准化，推荐）')

    parser.add_argument('--outlier_sigma_clip', type=float, default=None,
                       help='按训练集均值±N*std 将异常值置NaN（仅在提供standard_data_csv时推荐使用）')

    parser.add_argument('--depth_min', type=float, default=None,
                       help='仅对该深度以上数据做推理（ft）')
    parser.add_argument('--depth_max', type=float, default=None,
                       help='仅对该深度以下数据做推理（ft）')

    parser.add_argument('--kmax', type=int, default=32,
                       help='傅里叶模式数')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 配置
    config = {
        # 文件路径
        'las_file': args.las_file,
        'model_path': args.model_path,
        'output_dir': args.output_dir,
        'standard_data_csv': args.standard_data_csv,
        'outlier_sigma_clip': args.outlier_sigma_clip,
        'depth_min': args.depth_min,
        'depth_max': args.depth_max,

        # 数据预处理参数
        'window_size': 512,
        'stride': 256,

        # 模型参数
        'graph_channels': [512, 256, 128, 64, 32],
        'batch_size': 2,
        'input_features': ['GR', 'RHOB', 'NPHI', 'DTC'],
        'output_features': ['CALI', 'RSHA', 'RMED', 'RDEP', 'SP'],
        'head': 1,
        'drop': 0,
        'kmax': args.kmax,

        # 推理参数
        'inference_batch_size': 32,
    }

    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)

    # 运行流程
    pipeline = KGSInferencePipeline(config)
    results = pipeline.run()

    return results


if __name__ == "__main__":
    results = main()
