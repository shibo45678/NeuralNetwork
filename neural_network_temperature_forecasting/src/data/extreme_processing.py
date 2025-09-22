import pandas as pd
import numpy as np
from ..data.processing import DataPreprocessor
from scipy import stats


class ExtremeDataHandler(DataPreprocessor):  # 继承
    """极端数据专用处理"""

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)  # 继承父类构造函数

    def custom_handler(self)->'ExtremeDataHandler':
        """业务判断：物理上不可能的值 / 极端数据
           返回表格，根据索引处理"""

        """处理物理不可能值"""
        # 'wv'平均风速,'max. wv'最大风速列小于0，需将-9999 替换 为0，非删
        print(self.df[self.df['wv'] < 0]['wv'])
        print(self.df[self.df['max. wv'] < 0]['max. wv'])
        #
        self.df.loc[self.df['wv'] == -9999.0,'wv'] = 0
        self.df.loc[self.df['max. wv'] == -9999.0,'max. wv'] =0

        return self

    def single_remove_zscore(self, target_column: str = None, threshold: int = 3)->'ExtremeDataHandler':
        """使用Z-score方法移除[单列]异常值"""
        print(f"处理{target_column}列异常值(zscore)...")
        # super().identify_column_types() # 在"查看异常值"后，已有更新后的self.numeric_columns

        if target_column in self.numeric_columns:

            z_scores = np.abs(stats.zscore(self.df[target_column].dropna()))

            before_count = len(self.df)  # 从dropna()后算，仅算异常值数
            normal_mask = (z_scores < threshold)
            normal_indices = self.df[target_column].dropna().index[normal_mask]
            self.df = self.df.loc[normal_indices]
            after_count = len(self.df)

            if before_count != after_count:
                print(f"列'{target_column}':移除了 {before_count - after_count} 个异常值")
            else:
                print(f"列'{target_column}':无异常值可移除")

        return self

    def single_remove_iqr(self, target_column=None, threshold: float = 1.5)->'ExtremeDataHandler': # 不支持多特征联合异常处理
        """使用IQR方法移除异常值"""
        print(f"处理{target_column}列异常值(iqr)...")
        super().identify_column_types()
        if target_column in self.numeric_columns:
            if len(self.df[target_column].unique()) >= 4:
                Q1, Q3 = self.df[target_column].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                normal_mask = (self.df[target_column] >= lower_bound) & (self.df[target_column] <= upper_bound)

                before_count = len(self.df)
                self.df = self.df.loc[normal_mask]
                after_count = len(self.df)
                if before_count != after_count:
                    print(f"列'{target_column}':移除了 {before_count - after_count} 个异常值")
                else:
                    print(f"列'{target_column}':无异常值可移除")
            else:
                print(f"列'{target_column}':唯一值数量小于4，iqr计算不可靠")

        return self

    def get_handled_data(self)->pd.DataFrame:
        return self.df
    def analyze_extreme_patterns(self):
        """物理异常（错误数据）：物理可能性、时间持续性、空间一致性、设备状态、历史记录"""

        pass

    def assess_extreme_risks(self):
        pass
