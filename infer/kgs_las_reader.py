"""
KGS Well Data Reader and Preprocessor
从LAS文件中读取测井曲线数据，并进行预处理
"""

import numpy as np
import pandas as pd
import lasio
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class KGSLASReader:
    """读取KGS LAS文件并提取所需曲线"""

    def __init__(self, las_file_path: str):
        """
        初始化LAS读取器

        Args:
            las_file_path: LAS文件路径
        """
        self.las_file_path = las_file_path
        self.las = None
        self.df = None
        self.null_value = -999.25

    def read_las(self) -> pd.DataFrame:
        """
        读取LAS文件

        Returns:
            包含所有测井曲线的DataFrame
        """
        print(f"\n正在读取LAS文件: {self.las_file_path}")
        self.las = lasio.read(self.las_file_path)

        # 转换为DataFrame
        self.df = self.las.df()
        self.df.reset_index(inplace=True)

        print(f"成功读取LAS文件！")
        print(f"深度范围: {self.df['DEPT'].min():.2f} - {self.df['DEPT'].max():.2f} ft")
        print(f"采样间隔: {self.las.well.STEP.value} ft")
        print(f"可用曲线: {list(self.df.columns)}")

        return self.df

    def extract_curves(self, curve_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        提取并重命名所需曲线

        Args:
            curve_mapping: 曲线映射字典 {目标名称: LAS中的名称}
                例如: {'GR': 'GR', 'NPHI': 'NPHI', 'RHOB': 'RHOB',
                      'DTC': 'DELTAT', 'RSHA': 'RT30', 'RMED': 'RT60',
                      'RDEP': 'RT90', 'CALI': 'CALI', 'SP': 'SP'}

        Returns:
            包含提取和重命名后曲线的DataFrame
        """
        print(f"\n正在提取曲线...")
        extracted_data = {'DEPT': self.df['DEPT'].values}

        for target_name, source_name in curve_mapping.items():
            if source_name in self.df.columns:
                extracted_data[target_name] = self.df[source_name].values
                print(f"  ✓ {target_name} (从 {source_name} 提取)")
            else:
                print(f"  ✗ 警告: 未找到曲线 {source_name}")
                extracted_data[target_name] = np.full(len(self.df), np.nan)

        result_df = pd.DataFrame(extracted_data)
        return result_df

    def clean_data(self, df: pd.DataFrame, null_value: float = -999.25) -> pd.DataFrame:
        """
        清洗数据：将空值替换为NaN

        Args:
            df: 输入DataFrame
            null_value: LAS文件中的空值标记

        Returns:
            清洗后的DataFrame
        """
        print(f"\n正在清洗数据...")
        df_clean = df.copy()

        # 常见的LAS空值标记（不同供应商/导出工具可能不同）
        null_values = [
            null_value,
            -999.0,
            -9999.0,
            999.25,
            999.0,
            9999.0,
            1e30,
            -1e30,
        ]

        for col in df_clean.columns:
            if col != 'DEPT':
                # 将常见空值标记替换为NaN
                for nv in null_values:
                    df_clean.loc[np.isclose(df_clean[col], nv, equal_nan=False), col] = np.nan

        # 近似对齐训练数据清洗：把明显不合理的取值当作缺失
        # 目的：避免少量极端值把坐标轴拉得过大，并减少对标准化统计/插值的破坏
        physical_ranges = {
            'GR': (0.0, 400.0),
            'NPHI': (-0.5, 1.0),
            'RHOB': (1.0, 3.5),
            'DTC': (20.0, 250.0),
            'CALI': (0.0, 30.0),
            'SP': (-1000.0, 1000.0),
            # 电阻率使用对数坐标展示更合理，这里只做基本合法性约束
            'RSHA': (1e-6, np.inf),
            'RMED': (1e-6, np.inf),
            'RDEP': (1e-6, np.inf),
        }

        for col, (low, high) in physical_ranges.items():
            if col in df_clean.columns:
                invalid_mask = (~df_clean[col].isna()) & ((df_clean[col] < low) | (df_clean[col] > high))
                invalid_count = int(invalid_mask.sum())
                if invalid_count > 0:
                    df_clean.loc[invalid_mask, col] = np.nan
                    print(f"  {col}: 额外标记 {invalid_count} 个异常值为 NaN")

        for col in df_clean.columns:
            if col != 'DEPT':
                # 统计空值比例
                null_count = df_clean[col].isna().sum()
                null_pct = null_count / len(df_clean) * 100
                print(f"  {col}: {null_count} 个空值 ({null_pct:.2f}%)")

        return df_clean

    def remove_null_sections(self, df: pd.DataFrame,
                            required_curves: List[str]) -> pd.DataFrame:
        """
        移除所有必需曲线都有空值的深度段

        Args:
            df: 输入DataFrame
            required_curves: 必需的曲线列表（不包括DEPT）

        Returns:
            移除空值段后的DataFrame
        """
        print(f"\n正在移除全空值深度段...")
        original_len = len(df)

        # 创建掩码：至少有一个必需曲线有有效值
        valid_mask = df[required_curves].notna().any(axis=1)
        df_clean = df[valid_mask].copy()

        removed_count = original_len - len(df_clean)
        print(f"  移除了 {removed_count} 个深度点 ({removed_count/original_len*100:.2f}%)")
        print(f"  剩余深度范围: {df_clean['DEPT'].min():.2f} - {df_clean['DEPT'].max():.2f} ft")

        return df_clean.reset_index(drop=True)

    def interpolate_missing(self, df: pd.DataFrame,
                           curves: List[str],
                           method: str = 'linear',
                           limit: int = 20) -> pd.DataFrame:
        """
        插值填充缺失值

        Args:
            df: 输入DataFrame
            curves: 需要插值的曲线列表
            method: 插值方法 ('linear', 'polynomial', 'spline')
            limit: 最大插值点数

        Returns:
            插值后的DataFrame
        """
        print(f"\n正在插值填充缺失值 (方法: {method}, 最大连续插值: {limit} 点)...")
        df_interp = df.copy()

        for curve in curves:
            if curve in df_interp.columns and curve != 'DEPT':
                before_null = df_interp[curve].isna().sum()

                # 线性插值
                df_interp[curve] = df_interp[curve].interpolate(
                    method=method,
                    limit=limit,
                    limit_direction='both'
                )

                after_null = df_interp[curve].isna().sum()
                filled_count = before_null - after_null

                if filled_count > 0:
                    print(f"  {curve}: 填充了 {filled_count} 个缺失值")

        return df_interp

    def get_statistics(self, df: pd.DataFrame, curves: List[str]) -> pd.DataFrame:
        """
        计算曲线统计信息

        Args:
            df: 输入DataFrame
            curves: 曲线列表

        Returns:
            统计信息DataFrame
        """
        stats = {}
        for curve in curves:
            if curve in df.columns and curve != 'DEPT':
                valid_data = df[curve].dropna()
                stats[curve] = {
                    'mean': valid_data.mean(),
                    'std': valid_data.std(),
                    'min': valid_data.min(),
                    'max': valid_data.max(),
                    'median': valid_data.median(),
                    'count': len(valid_data)
                }

        return pd.DataFrame(stats).T


def main():
    """测试LAS读取器"""
    las_file = r"c:\Users\盖\Desktop\FNPG_JGRML\paper\KGS\1045063634.las"

    # 创建读取器
    reader = KGSLASReader(las_file)

    # 读取LAS文件
    df = reader.read_las()

    # 定义曲线映射（FNPG需要的曲线）
    # 输入曲线: GR, NPHI, RHOB, DTC
    # 输出曲线: CALI, RSHA, RMED, RDEP, SP
    curve_mapping = {
        # 输入特征
        'GR': 'GR',          # 自然伽马
        'NPHI': 'NPHI',      # 中子孔隙度
        'RHOB': 'RHOB',      # 密度
        'DTC': 'DELTAT',     # 声波时差

        # 输出特征（需要预测的）
        'CALI': 'CALI',      # 井径
        'RSHA': 'RT30',      # 浅电阻率 (30in)
        'RMED': 'RT60',      # 中电阻率 (60in)
        'RDEP': 'RT90',      # 深电阻率 (90in)
        'SP': 'SP'           # 自然电位
    }

    # 提取曲线
    df_extracted = reader.extract_curves(curve_mapping)

    # 清洗数据
    df_clean = reader.clean_data(df_extracted)

    # 移除全空值段
    input_curves = ['GR', 'NPHI', 'RHOB', 'DTC']
    df_valid = reader.remove_null_sections(df_clean, input_curves)

    # 插值填充
    all_curves = list(curve_mapping.keys())
    df_final = reader.interpolate_missing(df_valid, all_curves, limit=10)

    # 统计信息
    stats = reader.get_statistics(df_final, all_curves)
    print("\n" + "="*60)
    print("曲线统计信息:")
    print("="*60)
    print(stats.to_string())

    # 保存处理后的数据
    output_file = r"c:\Users\盖\Desktop\FNPG_JGRML\paper\KGS\1045063634_processed.csv"
    df_final.to_csv(output_file, index=False)
    print(f"\n已保存处理后的数据到: {output_file}")

    return df_final, stats


if __name__ == "__main__":
    df, stats = main()
