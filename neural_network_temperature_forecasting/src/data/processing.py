import pandas as pd
import numpy as np
from typing import Tuple, Union,List
from scipy import stats
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_df = df.copy()
        self.numeric_columns = None
        self.categorical_columns = None
        self.history = []  # 记录处理历史

    def identify_column_types(self):
        """识别数值型和分类型列"""
        # 数值型列(整型/浮点型）
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        # 分类型列(字符串/分类）
        self.categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        # 其他类型(日期/布尔等) 临时变量，不需要后续方法中频繁使用
        self.other_columns = self.df.select_dtypes(exclude=[np.number, 'object', 'category']).columns.tolist()

        print(f"数值型{len(self.numeric_columns)}列: {self.numeric_columns}")
        print(f"分类/字符串型{len(self.categorical_columns)}列: {self.categorical_columns}")
        print(f"其他类型{len(self.other_columns)}列: {self.other_columns}")

        self.history.append('识别类型')
        return self

    def process_numeric_data(self):
        """处理数值型数据"""
        print("处理数值型数据...")
        # 确认是数值型
        for col in self.numeric_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')  # 不报错，转NaN ，转整型 .astype('int64')
        print("确认是数值型列")

        self.history.append('处理数值型数据')
        return self

    def encode_categorical_data(self):
        """处理分类型/字符串数据"""
        print("处理分类型/字符串数据...")

        for col in self.categorical_columns:
            if col == 'Date Time':
                # 处理字符串时间 并排好序
                datetime = pd.to_datetime(self.df.pop(col), format='%d.%m.%Y %H:%M:%S')
                self.df[col] = datetime
                self.df = self.df.sort_values(col, ascending=True)
                print(f"处理时间字符串列{col},转成datetime格式")

            # 处理分类
            # 1.分类数量少，星期几月(独热编码)
            # 2.分类数量多，产品ID、店铺ID，模型内嵌入层 (Embedding Layer)，将高基数分类特征转换为密集向量表示
            # 即使输入已经处理，如果是预测分类变量，也要处理输出层激活函数以及损失函数。而且layers也是需要分开卷积再合并！
        self.history.append('处理分类型/字符串数据')
        return self

    def process_other_data(self):
        print("处理其他型(时间/布尔)数据...")
        if self.other_columns:
            other_df = self.df[self.other_columns]
        self.history.append('处理其他型(时间/布尔)数据')
        return self

    def handle_missing_values(self,
                              cat_strategy: str = 'custom',  # 支持众数填充/自定义Missing填充
                              num_strategy: str = 'mean', num_fill_value=None):  # 支持均值/众数/中位数/常数填充需写num_fill_value
        """处理缺失值"""
        print("==========统计空值结果==========")
        print(self.df.isna().sum())

        print("处理缺失值...")
        """1.分类列/字符列填充"""
        for col in self.categorical_columns:
            if self.df[col].isna().any():
                # 众数填充
                if cat_strategy == 'mode':
                    # 确保列中有非空值来计算众数
                    non_null_data = self.df[col].dropna()
                    if len(non_null_data) > 0:
                        mode_val = non_null_data.mode()  # 多个众数
                        if len(mode_val) > 0:
                            self.df[col].fillna(mode_val[0], inplace=True)
                            print(f"categorical:{col}列，{cat_strategy}填充模式完成填充(第1个众数)")
                        else:
                            self.df[col].fillna('Unknown', inplace=True)
                            print(f"categorical:{col}列，{cat_strategy}填充模式完成填充(仅1个众数)")
                    else:  # 整列空值，填充Missing
                        self.df[col].fillna('Missing', inplace=True)
                        print(f"categorical:{col}列，{cat_strategy}填充模式无法填充(整列空值)")

                # 自定义Missing
                if cat_strategy == 'custom':
                    self.df[col].fillna('Missing', inplace=True)
                    print(f"categorical:{col}列，'自定义'填充模式(保留Missing)")

        """2.数值列填充"""
        for col in self.numeric_columns:
            if self.df[col].isna().sum() > 0:
                if num_strategy == 'mean':
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                    print(f"numeric:{col}列，{num_strategy}填充模式完成填充，填充值{self.df[col].mean()}")
                elif num_strategy == 'median':
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                    print(f"numeric:{col}列，{num_strategy}填充模式完成填充，填充值{self.df[col].median()}")
                elif num_strategy == 'mode':  # 第一个众数
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                    print(f"numeric:{col}列，{num_strategy}填充模式完成填充，填充值{self.df[col].mode()[0]}")
                elif num_strategy == 'constant' and num_fill_value is not None:
                    self.df[col].fillna(num_fill_value, inplace=True)
                    print(f"numeric:{col}列，{num_strategy}填充模式完成填充，填充值{num_fill_value}")

        self.history.append('处理缺失值')
        return self

    def remove_duplicates(self):
        """移除重复行"""
        print("移除重复行...")
        df = self.df.copy()
        # 所有重复的行都为True，只有唯一的行为False,默认'first'是False被保留
        duplicate_mask = df.duplicated(keep=False)
        duplicate_rows = df[duplicate_mask]

        initial_count = len(self.df)
        self.df.drop_duplicates(inplace=True)
        removed_count = initial_count - len(self.df)
        print(f"移除了{removed_count}个重复行")

        self.history.append("处理重复行")
        return self

    def delete_useless_cols(self, target_cols: list = None):
        """移除无用列"""
        print("移除无用列...")
        self.history.append("移除无用列")
        return self

    def create_extreme_features_zscore(self, threshold: int = 3):  # z = (x - μ) / σ 单位标准差 >=3个标准差算异常
        self.identify_column_types()
        """使用Z-score方法标记每列异常值"""
        df = self.df.copy()
        print(f"检测数值列异常值(zscore)...")

        all_outliers_list = []
        for col in self.numeric_columns:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_mask = (z_scores >= threshold)
            # 获取异常值的索引
            outlier_indices = df[col].dropna().index[outlier_mask]

            if len(outlier_indices) > 0:
                outlier_df = (df.loc[outlier_indices].copy()
                              .assign(outlier_source=col,
                                      z_score=z_scores[outlier_mask],
                                      original_index=outlier_indices))

                all_outliers_list.append(outlier_df)
                print(f"列'{col}':检测到{len(outlier_df)}个异常值")
            else:
                print(f"列'{col}':未检测到异常值")

        # 一次合并所有结果
        if all_outliers_list:
            all_outliers = pd.concat(all_outliers_list, ignore_index=True)

            # 一列多个异常结果合并
            result_df = (all_outliers.groupby(self.numeric_columns)
                         .agg(extreme_tag=('outlier_source', list),
                              abnormal_count=('outlier_source', 'count'),
                              original_index=('original_index', 'first'))
                         )
            result_df.to_csv("extreme_features_zscore.csv")
        else:
            all_outliers = pd.DataFrame()

        self.history.append("检测数值列异常值(zscore)")
        return self

    def create_extreme_features_iqr(self, threshold: float = 1.5):
        self.identify_column_types()
        """使用IQR方法标记每列异常值"""
        df = self.df.copy()
        print(f"检测数值列异常值(iqr)...")

        all_outliers_list = []
        for col in self.numeric_columns:

            if len(df[col].unique()) >= 4:
                Q1, Q3 = self.df[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_indices = df[col].index[outlier_mask]

                if outlier_mask.sum() > 0:
                    outlier_df = (df.loc[outlier_indices].copy()
                                  .assign(outlier_source=col,
                                          original_index=outlier_indices))  # 原始索引取出便于后续修改

                    all_outliers_list.append(outlier_df)
                    print(f"列'{col}':检测到{len(outlier_df)}个异常值")
                else:
                    print(f"列'{col}':未检测到异常值")

            else:
                print(f"列'{col}':唯一值样本数不足4个，IQR判断不适用，需要改用其他方法判断")

        # 一次合并所有结果
        if all_outliers_list:
            all_outliers = pd.concat(all_outliers_list, ignore_index=True)

            # 一列多个异常结果合并
            result_df = (all_outliers.groupby(self.numeric_columns)
                         .agg(extreme_tag=('outlier_source', list),
                              abnormal_count=('outlier_source', 'count'),
                              original_index=('original_index', 'first')))
            result_df.to_csv("extreme_features_iqr.csv")
        else:
            all_outliers = pd.DataFrame()

        self.history.append("检测数值列异常值(iqr)")
        return self

    def create_extreme_features_multivariate(self, contamination=0.025):  # 预期异常比例 ≈2.5%
        self.identify_column_types()
        """多变量联合异常检测
           多变量联合分析，不是逐列处理
           某个点可能单个特征正常，但多个特征的组合异常
        """
        df = self.df.copy()
        print(f"检测联合异常值(iso_forest)...")

        from sklearn.ensemble import IsolationForest
        # 1.使用隔离森林检测整体异常
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        outliers = iso_forest.fit_predict(df[self.numeric_columns])

        # 2.标记异常点
        df['is_outlier'] = outliers == -1
        print(f"检测到{df['is_outlier'].sum()}个多变量异常点")
        outliers_indices = df.index[outliers == -1]
        result_df = df.loc[outliers_indices]
        result_df.to_csv("extreme_features_isoforest.csv")

        self.history.append("检测数值列异常值(iso_forest)")
        return self

    def get_history(self):
        return self.history  # 查看清理历史

    def get_summary(self):
        """获取处理摘要"""
        original_shape = self.original_df.shape
        processed_shape = self.df.shape

        print(f"原始数据形状：{original_shape}")
        print(f"处理后数据形状：{processed_shape}")
        print(f"移除了 {original_shape[0] - processed_shape[0]} 行")
        return self

    def get_processed_data(self):
        return self.df.copy()

    # 通过方法暴露数据
    def get_numeric_columns(self):
        """获取数值型列名"""
        if self.numeric_columns is None:
            self.identify_column_types()
        return self.numeric_columns.copy()  # 返回副本避免外部修改

    def get_categorical_columns(self):
        if self.categorical_columns is None:
            self.identify_column_types()
        return self.categorical_columns.copy()


class DataResampler:
    """数据重采样"""

    def __init__(self, data: pd.DataFrame):
        self.original_data = data.copy()
        self.resampled_data = None

    def systematic_resample(self,
                            start_index: int = 0,
                            step: int = 1) -> 'DataResampler':
        """系统抽样（等间隔抽样）"""
        # 保证数据是排好序的
        self.resampled_data = self.original_data.iloc[start_index::step]
        print(f"等间隔抽样: 从索引 {start_index} 开始，步长 {step}，共 {len(self.resampled_data)} 个样本")

        return self  # 返回实例本身以支持链式调用

    def time_based_resample(
            self,
            time_column: str = None,
            freq: str = 'H',  # 重采样频率 ('H'-小时, 'D'-天, 'W'-周等)
            aggregation: str = 'mean'  # 聚合方法 ('mean', 'sum', 'max', 'min', 'first', 'last')
    ) -> 'DataResampler':
        """基于时间的重采样（适用于时间序列数据）"""
        if time_column not in self.original_data.columns:
            raise ValueError(f"时间列 '{time_column}' 不存在")

        self.resampled_data = (
            self.original_data
            .set_index(time_column)
            .resample(freq)
            .agg(aggregation)
            .reset_index()
        )
        print(f"时间重采样: 频率 {freq}，聚合方法 {aggregation}")

        return self

    def get_summary(self):
        """获取处理摘要"""
        original_shape = self.original_data.shape
        resampled_shape = self.resampled_data.shape

        print(f"原始数据形状：{original_shape}")
        print(f"重采样后数据形状：{resampled_shape}")
        print(f"移除了 {original_shape[0] - resampled_shape[0]} 行")
        return self

    def get_resampled_data(self) -> pd.DataFrame:
        """上面需要支持链式调用"""
        return self.resampled_data.copy()


class DataSplitter:
    def __init__(self, df:pd.DataFrame):
        self.df = df
        self.trainSets =pd.DataFrame()
        self.valSets = pd.DataFrame()
        self.testSets = pd.DataFrame()
        self.scalers = {}  # 初始化标准化器字典


    def train_val_test_split(self,
                             train_size: float = 0.7,
                             val_size: float = 0.2,
                             test_size: float = 0.1) -> 'DataSplitter':

        """时间序列的分层，按顺序整体切3部分"""
        n = len(self.df)
        self.trainSets = self.df.iloc[0:int(n * train_size)]
        self.valSets = self.df.iloc[int(n * train_size):int(n * (train_size + val_size))]
        self.testSets = self.df.iloc[int(n * (train_size + val_size)):]

        print(
            f"数据分割完成: 训练集 {len(self.trainSets)} 样本, 验证集 {len(self.valSets)} 样本,测试集 {len(self.testSets)} 样本")
        return self

    def has_validation_set(self) -> bool:
        """检查是否有验证集"""
        return not self.valSets.empty and len(self.valSets) > 0

    def standardize_data(self) -> 'DataSplitter':  # 即zscore（原值-均值）/ 标准差
        """Z_score标准化（使用训练集统计量）"""
        print("数据标准化(zscore)...")

        for col in self.df.columns:

            if self.trainSets[col].notna().sum() > 1:  # 至少2个非空
                train_col = self.trainSets[col].dropna()
                mean_val = train_col.mean()  # 防止数据泄漏 只用训练集的均值和标准差
                std_val = train_col.std()

                if std_val > 1e-8:  # 避免除零错误
                    self.trainSets.loc[:,col] = (self.trainSets[col] - mean_val) / std_val

                    if self.has_validation_set() and col in self.valSets.columns:
                        self.valSets.loc[:,col] = (self.valSets[col] - mean_val) / std_val

                    if not self.testSets.empty and col in self.testSets.columns:
                        self.testSets.loc[:,col] = (self.testSets[col] - mean_val) / std_val

                    self.scalers[col] = {'type': 'zscore', 'mean': mean_val, 'std': std_val, 'method': 'standardize'}
                    print(f"列 {col}: Z-score 标准化完成")

                else:
                    # 处理零标准差情况
                    print(f"列{col} 标准差为0 ，跳过标准化")
                    self.trainSets[col] = 0  # 所有值相同，设为0
                    if self.has_validation_set() and col in self.valSets.columns:
                        self.valSets[col] = 0
                    if not self.testSets.empty and col in self.testSets.columns:
                        self.testSets[col] = 0
                    print(f"列 {col}: 标准差为0，设为常数0")

        return self

    def normalize_data(self) -> 'DataSplitter':  # 归一化 (Normalization)	(x - min) / (max - min)
        """Min-Max 归一化（使用训练集统计量）"""
        print("数据归一化(min_max)...")
        for col in self.df.columns:

            if self.trainSets[col].notna().sum() > 1:
                train_col = self.trainSets[col].dropna()
                min_val = train_col.min()
                max_val = train_col.max()

                if max_val > min_val:
                    self.trainSets.loc[:,col] = (self.trainSets[col] - min_val) / (max_val - min_val)
                    print(f"训练集列{col}:min_max 归一化完成")

                    if self.has_validation_set() and col in self.valSets.columns:
                        self.valSets.loc[:,col] = (self.valSets[col] - min_val) / (max_val - min_val)
                        print(f"验证集列{col}:min_max 归一化完成")

                    if not self.testSets.empty and col in self.testSets.columns:
                        self.testSets.loc[:,col] = (self.testSets[col] - min_val) / (max_val - min_val)
                        print(f"测试集列{col}:min_max 归一化完成")

                    self.scalers[col] = {'type': 'minmax', 'min': min_val, 'max': max_val, 'method': 'normalize'}
                    print(f"列 {col}: 归一化完成")

                else:  # 所有值相同的情况
                    self.trainSets[col] = 0
                    if self.has_validation_set() and col in self.valSets.columns:
                        self.valSets[col] = 0
                    if not self.testSets.empty and col in self.testSets.columns:
                        self.testSets[col] = 0
                    print(f"列 '{col}': 最大值等于最小值，设为0")

        return self

    def get_transformed_data(self) -> Tuple:
        self.df = pd.concat([self.trainSets, self.valSets, self.testSets], ignore_index=False)

        return self.df.copy(),self.trainSets.copy(), self.valSets.copy(), self.testSets.copy()

    def get_scalers(self) -> dict:
        return self.scalers.copy()
