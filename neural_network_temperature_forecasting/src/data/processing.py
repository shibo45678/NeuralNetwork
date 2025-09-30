from sklearn.base import BaseEstimator, TransformerMixin

from pathlib import Path
import codecs
import os
import csv
import glob
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from scipy import stats
import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from ..data.exploration import Visualization
import joblib


class DataPreprocessor:
    def __init__(self, input_files: list):


        self.trainSets = pd.DataFrame()
        self.valSets = pd.DataFrame()
        self.testSets = pd.DataFrame()
        self.scalers = {}  # 初始化标准化器字典
        self.constant_values = {}
        self.history = None


"""加载文件"""


class DataLoader(BaseEstimator, TransformerMixin):
    def __init__(self, input_files: List[str], pattern: str, data_dir: str = "data"):
        self.input_files = input_files
        self.pattern = pattern
        self.data_dir = data_dir
        self.dir_path = None
        self.merged_df = None
        self.history = []



    def fit(self, X=None, y=None) -> 'DataLoader':
        PROJECT_ROOT = Path(__file__).parent.parent.parent
        self.dir_path = PROJECT_ROOT / self.data_dir
        self.history.append("初始化数据处理器")
        return self

    def transform(self, X=None) -> pd.DataFrame:
        # 1. 编码转换
        self._handle_encoding()
        # 2. 数据加载和合并
        self._load_all_data()
        # 3. 保存原始数据
        self.origin_df = self.merged_df.copy()
        self.history.append("完成数据加载和合并")
        return self.merged_df

    def _handle_encoding(self):
        # 1.指定文件所在目录 - 根据源文件名进行命名"new_"
        # 2.创建空csv文件（拼接路径 os.path.join() - 写入表头）
        newfile_names = []
        for filename in self.input_files:
            newfile_name = "new_" + filename
            newfile_names.append(newfile_name)

            original_file = os.path.join(self.dir_path, filename)
            new_file = os.path.join(self.dir_path, filename)

            # 具体的编码转换实现
            self._convert_file_encoding(original_file, new_file)
        self.history.append("完成文件编码转换")

    """单个文件的编码转换"""

    def _convert_file_encoding(self, original_file: str, new_file: str):
        # 创建csv写入器
        with open(new_file, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)  # writer.writerow(["",""]) # 写入表头

        # 按照确定的 encoding 读取旧文件内容，另存为utf-8编码内容的新文件
        f = open(original_file, "rb+")
        content = f.read()  # 读取文件内容，content为bytes类型，而非string类型
        source_encoding = "utf-8"  # 初始化source_encoding

        try:
            # 尝试以不同的编码解码内容
            for encoding in ["utf-8", "gbk", "gb2313", "gb18030", "big5", "cp936"]:
                try:
                    decode_content = content.decode(encoding)
                    source_encoding = encoding
                    break  # 如果找到匹配的编码，就跳出循环
                except UnicodeDecodeError:
                    pass  # 如果解码失败，继续尝试其他编码
            else:  # 如果循环结束还没有找到匹配的编码
                print("无法确定原始编码")

        except Exception as e:
            print(f"发生错误：{e}")
        finally:
            f.close()  # 确保文件总是关闭的

        # 编码：读取-存取
        block_size = 4096
        with codecs.open(original_file, "r", source_encoding) as f:
            with codecs.open(new_file, "w", "utf-8") as f2:
                while True:
                    content = f.read(block_size)
                    if not content:
                        break
                    f2.write(content)

    # 加载和合并所有数据
    def _load_all_data(self):

        full_pattern = os.path.join(self.dir_path, self.pattern)
        all_files = glob.glob(full_pattern)  # 获取解析后的文件

        print(f"搜索模式：{full_pattern}")
        print(f"找到的文件：{all_files}")

        data_frames = []
        for file_path in all_files:
            try:
                df = pd.read_csv(file_path)
                data_frames.append(df)
                print(f"成功读取: {file_path}, 形状: {df.shape}, dtype:{df.dtypes}")
            except Exception as e:
                print(f"读取文件失败{file_path}:{str(e)}")

        if data_frames:
            self.merged_df = pd.concat(data_frames, ignore_index=True)

            # 保存合并后的文件
            output_path = os.path.join(self.dir_path, "merged_data.csv")
            self.merged_df.to_csv(output_path, index=False)
            print(f"合并文件将保存到: {output_path}")

        else:
            raise ValueError(f"没有找到匹配的文件")


"""描述性分析"""


class DescribeData(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stats = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            self._describe_data(X)
        return X

    def _describe_data(self, df: pd.DataFrame):
        self.stats = df.describe()
        print("描述性分析如下：")
        print(self.stats)

        print("\n 数据类型：")
        print(df.dtypes)

        print("\n 缺失值统计：")
        print(df.isna().sum())


"""一般问题列正则处理"""


class ProblemColumnsFixed(BaseEstimator, TransformerMixin):

    def __init__(self, problem_columns: list = None):
        self.problem_columns = problem_columns
        self.columns_to_process_ = []

    def fit(self, X, y=None):
        print("一般问题列正则处理...")
        """验证问题列是否存在"""
        df = pd.DataFrame(X)
        if self.problem_columns is None:
            print("使用修复列功能，但未指定待修复问题列")
            return self

        # 确认问题列是否存在
        self.columns_to_process_ = [col for col in self.problem_columns if col in df.columns]
        missing_cols = [col for col in self.problem_columns if col not in df.columns]

        if missing_cols:
            print(f"以下列不存在，将跳过: {missing_cols}")

        print(f"将处理 {len(self.columns_to_process_)} 个问题列: {self.columns_to_process_}")
        return self

    def transform(self, X):
        """应用正则清洗转换"""
        if not self.columns_to_process_:
            return X
        df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        processed_count = 0

        for col in self.columns_to_process_:
            if col in df.columns:
                # 记录处理前信息
                non_null_count = df[col].notna().sum()
                if non_null_count > 0:
                    sample_value = df[col].iloc[0]
                    print(f"问题列第一个元素：{sample_value}")

                    df[col] = (df[col]
                    .astype(str)
                    .str.extract(r'([-+]?\d*\.?\d+)', expand=False)[0])  # expand=False 返回Series

                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    processed_count += 1
                    print(f"列 '{col}':已转文本，正则清洗，转回数值")
        print(f"正则清洗完成: 成功处理 {processed_count}/{len(self.columns_to_process_)} 个列")
        return df


"""修复问题列-列包含df"""


class SpecialColumnsFixed(BaseEstimator, TransformerMixin):
    def __init__(self, problem_columns: list = None):
        self.problem_columns = problem_columns
        self.columns_to_process_ = []

    def fit(self, X, y=None):
        if self.problem_columns is None:
            print("使用'特别修复'列功能，但未指定待修复问题列")
            return self

        df = pd.DataFrame(X)

        missing_cols = [col for col in self.problem_columns if col not in df.columns]
        if missing_cols:
            print(f"以下列不存在，将跳过: {missing_cols}")

        for col in self.problem_columns:
            if col in df.columns:
                # 检查Series内部结构
                print(f"Series 类型:{type(df[col])}")
                print(f"Series dtype:{df[col].dtype}")
                print(f"Series形状:{df[col].shape}")

                # 提取
                series = df[col].copy()

                # 判断是否需要处理
                needs_processing = False
                skip_further_checks = False

                # 一、检查第一个非空值的实际类型,是否是DataFrame
                first_check = series.dropna(inplace=False)
                first_non_null = first_check.iloc[0] if not first_check.empty else None
                print(f"问题列{col}的第一个元素的dtype:{type(first_non_null)}")

                if isinstance(first_non_null, pd.DataFrame):
                    print(f"确认{col}列包含DataFrame对象，第一个元素形状：{first_non_null.shape}")
                    needs_processing = True
                    skip_further_checks = True  # 标记跳过后续检查

                elif first_non_null is not None and hasattr(first_non_null, 'iloc'):
                    print(f"列 '{col}' 第一个值有iloc方法，可能是特殊对象")
                    needs_processing = True
                    skip_further_checks = True
                    try:
                        # 尝试提取值
                        if hasattr(first_non_null, 'shape') and first_non_null.shape[1] > 0:
                            inner_value = first_non_null.iloc[0, 0]
                        else:
                            inner_value = first_non_null.iloc[0]
                        print(f"内部值: {inner_value} (类型: {type(inner_value)})")
                    except Exception as e:
                        print(f"无法访问内部值: {e}")

                if not skip_further_checks:  # 第一个值检查完，检查整列是否有DataFrame
                    try:
                        has_other_dataframe = any(
                            isinstance(inner_value, pd.DataFrame) for inner_value in series if pd.notna(inner_value))
                        if has_other_dataframe:
                            print(f"{col}列中有其他单元格包含DataFrame对象")
                            needs_processing = True
                    except Exception as e:
                        print(f"检查整列DataFrame时出错: {e}")

                # 最终判断：如果列确实有问题，需要处理 添加
                if needs_processing:
                    self.columns_to_process_.append(col)
                    print(f"添加列 '{col}' 到处理列表")

        print(f"将处理 {len(self.columns_to_process_)} 个问题列: {self.columns_to_process_}")

        return self

    def transform(self, X):
        print("开始修改问题列...")
        if not self.columns_to_process_:
            print("没有需要修复的列")
            return X

        df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()

        for col in self.columns_to_process_:  # fit判断过，但可能单独调transform
            if col not in df.columns:
                print(f"列 '{col}' 不存在于数据中，跳过")
                continue

            print(f"\n修复列: {col}")
            series = df[col].copy()
            # 提取每个DataFrame的第一个值
            extracted_values = []
            for i, inner_value in enumerate(series):
                if pd.isna(inner_value):
                    # 保持NaN值
                    extracted_values.append(np.nan)

                elif isinstance(inner_value, pd.DataFrame) and not inner_value.empty:
                    # 提取第一个单元格的值
                    try:
                        if inner_value.shape[1] > 0:
                            extracted_values = inner_value.iloc[0, 0]
                        else:
                            extracted_values = inner_value.iloc[0]
                        extracted_values.append(extracted_values)
                        print(f"索引 {i}: DataFrame -> {extracted_values}")

                    except Exception as e:
                        print(f" 索引 {i}: 提取失败 -> {e}")
                        extracted_values.append(np.nan)
                else:
                    extracted_values.append(inner_value)

            # 创建新的Series
            series_fixed = pd.Series(extracted_values, index=series.index, name=col)

            # 替换原列
            df[col] = series_fixed

            # 验证修复结果
            if len(df[col]) > 0:
                sample_value = df[col].iloc[0]
                print(f"修复后的{col}列类型: {type(sample_value)},值: {sample_value}")

            """检查原Series是否在某些操作下表现出DataFrame行为"""
            print("\n=== 行为测试 ===")
            # 测试1：尝试转置
            try:
                transposed = series.transpose()
                print(f"转置结果类型：{type(transposed)}")
                if hasattr(transposed, 'shape'):
                    print(f"转置形状：{transposed.shape}")
            except Exception as e:
                print(f"转置失败：{str(e)}")

            # 测试2：尝试访问列
            try:
                if hasattr(series, 'columns'):
                    print(f"有columns属性：{series.columns}")
                else:
                    print("没有columns属性")
            except Exception as e:
                print(f"检查columns失败：{str(e)}")

        print(f"\n修复完成: 处理了 {len(self.columns_to_process_)} 个问题列")
        return df


"""识别列类型"""


class ColumnsTypeIdentify(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numeric_columns = None
        self.categorical_columns = None
        self.other_columns = None

    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        # 数值型列(整型/浮点型）
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # 分类型列(字符串/分类）
        self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        # 其他类型(日期/布尔等) 临时变量，不需要后续方法中频繁使用
        self.other_columns = df.select_dtypes(exclude=[np.number, 'object', 'category']).columns.tolist()

        print(f"数值型{len(self.numeric_columns)}列: {self.numeric_columns}")
        print(f"分类/字符串型{len(self.categorical_columns)}列: {self.categorical_columns}")
        print(f"其他类型{len(self.other_columns)}列: {self.other_columns}")

        return self

    def transform(self, X):
        return X


"""处理数值型数据"""


class ProcessNumericColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols: Optional[list] = None,preserve_integer_types:bool=True):
        self.numeric_columns = cols
        # object(字符串/混合类型，里面'1', '2', 'abc'] -> [1.0, 2.0, nan]) 默认是float64 -> 改为 [1, 2, nan]
        # 其他数值型不变 int64，float64
        self.preserve_integer_types=preserve_integer_types
        self.original_dtypes_ ={}

    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        if self.numeric_columns is None:  # None ，空列表[]
            print("未指定待处理数值列，检查数据全部数值列")
            self.numeric_columns = df.select_dtypes(include=[np.number]).columns.to_list()
            if not self.numeric_columns:
                print("数据无数值列需处理")
            else:
                print(f"自动识别数值列: {self.numeric_columns}")
        else:
            # 检查指定的列是否实际存在
            existing_cols = [col for col in self.numeric_columns if col in df.columns]
            missing_cols = [col for col in self.numeric_columns if col not in df.columns]

            if missing_cols:
                print(f"警告: 以下指定列不存在: {missing_cols}")

            self.numeric_columns = existing_cols
            print(f"使用指定数值列: {self.numeric_columns}")

            # 记录原始数据类型
            for col in self.numeric_columns:
                if col in df.columns:
                    self.original_dtypes_[col] = df[col].dtype

        return self

    def transform(self, X):
        print("处理数值型数据...")
        df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()

        if not self.numeric_columns:
            print("无数值列需要处理")
            return df

        for col in self.numeric_columns:
            if col in df.columns:
                # 保存原类型
                original_dtype = df[col].dtypes

                df[col] = pd.to_numeric(df[col], errors='coerce')  # object 不报错，转NaN 默认是float64。

                # 如果标记为保持整数类型、原始是整数类型、转换后没有小数部分，尝试转回整数
                if (self.preserve_integer_types and
                        col in self.original_dtypes_ and
                        np.issubdtype(self.original_dtypes_[col], np.integer)):

                    # 检查是否所有非空值都是整数
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        # 方法：直接检查小数部分是否为0 .00
                        decimal_parts = non_null_values % 1
                        all_integers = np.all(decimal_parts == 0) # bool
                        if all_integers:
                            df[col] = df[col].astype('Int64')
                print(f"列 {col} 已确认是数值型 (原类型: {original_dtype} -> 现类型: {df[col].dtype})")
            else:
                print(f"列{col}不在数据中")
                continue

        print("数值型数据处理完成")
        return df

    """处理分类型/字符串数据"""

    def encode_categorical_data(self):
        """处理分类型/字符串数据"""
        print("处理分类型/字符串数据...")
        if self.categorical_columns in None:
            print("无分类型/字符串型列不需要处理")
            return self
        else:
            for col in self.categorical_columns:
                if col == 'Date Time':
                    # 处理字符串时间 并排好序
                    datetime = pd.to_datetime(self.origin_df.pop(col), format='%d.%m.%Y %H:%M:%S')
                    self.origin_df[col] = datetime
                    self.origin_df = self.origin_df.sort_values(col, ascending=True)
                    print(f"已处理时间字符串列{col}，转成datetime格式")

                # 处理分类
                # 1.分类数量少，星期几月(独热编码)
                # 2.分类数量多，产品ID、店铺ID，模型内嵌入层 (Embedding Layer)，将高基数分类特征转换为密集向量表示
                # 即使输入已经处理，如果是预测分类变量，也要处理输出层激活函数以及损失函数。而且layers也是需要分开卷积再合并！
            self.history.append('处理分类型/字符串数据')
            return self

    """处理其他型(时间/布尔)数据"""

    def process_other_data(self):
        print("处理其他型(时间/布尔)数据...")
        if self.other_columns is None:
            print("无其他型(时间/布尔)不需要处理")
            return self
        else:
            other_df = self.origin_df[self.other_columns]
            self.history.append('处理其他型(时间/布尔)数据')

        return self


def process_numeric_data(self):
    """处理数值型数据"""
    print("处理数值型数据...")
    if self.numeric_columns in None:
        print("无数值列不需要处理")
        return self
    else:
        # 确认是数值型
        for col in self.numeric_columns:
            self.origin_df[col] = pd.to_numeric(self.origin_df[col],
                                                errors='coerce')  # 不报错，转NaN ，转整型 .astype('int64')
        print("数值列已确认是数值型")
        self.history.append('处理数值型数据')

    return self


"""处理分类型/字符串数据"""


def encode_categorical_data(self):
    """处理分类型/字符串数据"""
    print("处理分类型/字符串数据...")
    if self.categorical_columns in None:
        print("无分类型/字符串型列不需要处理")
        return self
    else:
        for col in self.categorical_columns:
            if col == 'Date Time':
                # 处理字符串时间 并排好序
                datetime = pd.to_datetime(self.origin_df.pop(col), format='%d.%m.%Y %H:%M:%S')
                self.origin_df[col] = datetime
                self.origin_df = self.origin_df.sort_values(col, ascending=True)
                print(f"已处理时间字符串列{col}，转成datetime格式")

            # 处理分类
            # 1.分类数量少，星期几月(独热编码)
            # 2.分类数量多，产品ID、店铺ID，模型内嵌入层 (Embedding Layer)，将高基数分类特征转换为密集向量表示
            # 即使输入已经处理，如果是预测分类变量，也要处理输出层激活函数以及损失函数。而且layers也是需要分开卷积再合并！
        self.history.append('处理分类型/字符串数据')
        return self


"""处理其他型(时间/布尔)数据"""


def process_other_data(self):
    print("处理其他型(时间/布尔)数据...")
    if self.other_columns is None:
        print("无其他型(时间/布尔)不需要处理")
        return self
    else:
        other_df = self.origin_df[self.other_columns]
        self.history.append('处理其他型(时间/布尔)数据')

    return self


"""处理缺失值"""


def handle_missing_values(self,
                          cat_strategy: str = 'custom',  # 支持众数填充/自定义Missing填充
                          num_strategy: str = 'mean', num_fill_value=None):  # 支持均值/众数/中位数/常数填充需写num_fill_value
    """处理缺失值"""
    print("==========统计空值结果==========")
    print(self.origin_df.isna().sum())

    print("处理缺失值...")
    # 1.分类列/字符列填充
    for col in self.categorical_columns:
        if self.origin_df[col].isna().any():
            # 众数填充
            if cat_strategy == 'mode':
                # 确保列中有非空值来计算众数
                non_null_data = self.origin_df[col].dropna()
                if len(non_null_data) > 0:
                    mode_val = non_null_data.mode()  # 多个众数
                    if len(mode_val) > 0:
                        self.origin_df[col].fillna(mode_val[0], inplace=True)
                        print(f"categorical:{col}列，{cat_strategy}填充模式完成填充(第1个众数)")
                    else:
                        self.origin_df[col].fillna('Unknown', inplace=True)
                        print(f"categorical:{col}列，{cat_strategy}填充模式完成填充(仅1个众数)")
                else:  # 整列空值，填充Missing
                    self.origin_df[col].fillna('Missing', inplace=True)
                    print(f"categorical:{col}列，{cat_strategy}填充模式无法填充(整列空值)")

            # 自定义Missing
            if cat_strategy == 'custom':
                self.origin_df[col].fillna('Missing', inplace=True)
                print(f"categorical:{col}列，'自定义'填充模式(保留Missing)")

    # 2.数值列填充
    for col in self.numeric_columns:
        if self.origin_df[col].isna().sum() > 0:
            if num_strategy == 'mean':
                self.origin_df[col].fillna(self.origin_df[col].mean(), inplace=True)
                print(f"numeric:{col}列，{num_strategy}填充模式完成填充，填充值{self.origin_df[col].mean()}")
            elif num_strategy == 'median':
                self.origin_df[col].fillna(self.origin_df[col].median(), inplace=True)
                print(f"numeric:{col}列，{num_strategy}填充模式完成填充，填充值{self.origin_df[col].median()}")
            elif num_strategy == 'mode':  # 第一个众数
                self.origin_df[col].fillna(self.origin_df[col].mode()[0], inplace=True)
                print(f"numeric:{col}列，{num_strategy}填充模式完成填充，填充值{self.origin_df[col].mode()[0]}")
            elif num_strategy == 'constant' and num_fill_value is not None:
                self.origin_df[col].fillna(num_fill_value, inplace=True)
                print(f"numeric:{col}列，{num_strategy}填充模式完成填充，填充值{num_fill_value}")

    self.history.append('处理缺失值')
    return self


"""移除重复值"""


def remove_duplicates(self):
    """移除重复行"""
    print("移除重复行...")
    df = self.origin_df.copy()
    # 所有重复的行都为True，只有唯一的行为False,默认'first'是False被保留
    duplicate_mask = df.duplicated(keep=False)
    duplicate_rows = df[duplicate_mask]
    # duplicate_rows.to_csv("duplicate_rows.csv") # 下载重复数据

    initial_count = len(self.origin_df)
    self.origin_df.drop_duplicates(inplace=True)
    removed_count = initial_count - len(self.origin_df)
    print(f"移除了{removed_count}个重复行")

    self.history.append("处理重复行")
    return self


"""删除无用列"""


def delete_useless_cols(self, target_cols: list = None):
    """移除无用列"""
    print("移除无用列...")
    if target_cols is None:
        print("调用删除无用列功能，但未填写列名")
        return self
    else:
        self.origin_df.drop(target_cols, axis=1)
        print(f"移除了{len(target_cols)}个列")
        self.history.append("移除无用列")
    return self


"""查看数值列异常值(3种方式)"""


def check_extreme_features(self, method: Dict = None):  # z = (x - μ) / σ 单位标准差 >=3个标准差算异常
    print("查看数值列异常值...")
    """使用IQR或Z-score或ISO方法标记异常值
        method : Dict, optional
            检测方法配置，默认使用Z-score
            Example: or {'name': 'zscore', 'threshold': 3}
                     or {'name':'zscore','threshold':3}
                     or {'name':'multivariate','contamination':0.025}
    """
    if method is None:
        print("不查看异常值")

    if method['name'] == 'zscore':  # {'name':'zscore','threshold':3}
        """使用Z-score方法标记每列异常值"""
        df = self.origin_df.copy()
        print(f"检测数值列异常值(zscore)...")

        all_outliers_list = []
        for col in self.numeric_columns:
            clean_series = df[col].dropna()
            z_scores = np.abs(stats.zscore(clean_series))
            outlier_mask = (z_scores >= method['threshold'])
            # 获取异常值的索引
            outlier_indices = clean_series.index[outlier_mask]  # 保持一致 dropna

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
                         .agg(extreme_tag=('outlier_source', list),  # 按照所有列聚合后，统计某行数据的异常来源
                              abnormal_count=('outlier_source', 'count'),
                              original_index=('original_index', 'first'))
                         )
            # result_df.to_csv("extreme_features_zscore.csv")
        else:
            all_outliers = pd.DataFrame()
            print("所有列都未检查到异常值zscore")

        self.history.append("检测数值列异常值(zscore)")
        return self

    if method['name'] == 'iqr':
        """使用IQR方法标记每列异常值"""
        df = self.origin_df.copy()
        print(f"检测数值列异常值(iqr)...")

        all_outliers_list = []
        for col in self.numeric_columns:
            clean_series = df[col].dropna()
            if len(clean_series) >= 4 and len(clean_series.unique()) >= 4:  # 默认inplace=False
                try:
                    Q1, Q3 = clean_series.quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    if IQR == 0:  # 所有值相同
                        print(f"警告: 列 {col} 的IQR为0，可能所有值都相同")
                        continue

                    lower_bound = Q1 - method['threshold'] * IQR
                    upper_bound = Q3 + method['threshold'] * IQR
                    outlier_mask = (clean_series < lower_bound) | (clean_series > upper_bound)  # 或
                    outlier_indices = clean_series.index[outlier_mask]

                    if np.any(outlier_mask):  # sum() > 0 非最语义化,性能差，需要计算所有值的和
                        outlier_df = (df.loc[outlier_indices].copy()
                                      .assign(outlier_source=col,
                                              original_index=outlier_indices))  # 原始索引取出便于后续修改

                        all_outliers_list.append(outlier_df)
                        print(f"列'{col}':检测到{len(outlier_df)}个异常值")
                    else:
                        print(f"列'{col}':未检测到异常值")

                except Exception as e:
                    print(f"计算列 {col} 的IQR时出错: {e}")

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
            # result_df.to_csv("extreme_features_iqr.csv")
        else:
            all_outliers = pd.DataFrame()

        self.history.append("检测数值列异常值(iqr)")
        return self

    if method['name'] == 'multivariate':  # {'name':'multivariate','contamination':0.025}
        """多变量联合异常检测
           多变量联合分析，不是逐列处理
           某个点可能单个特征正常，但多个特征的组合异常"""
        df = self.origin_df.copy()
        print(f"检测联合异常值(iso_forest)...")

        from sklearn.ensemble import IsolationForest
        # 1.使用隔离森林检测整体异常
        iso_forest = IsolationForest(
            contamination=method['contamination'],  # 预期异常比例 ≈2.5%
            random_state=42
        )
        outliers = iso_forest.fit_predict(df[self.numeric_columns])

        # 2.标记异常点
        df['is_outlier'] = outliers == -1
        print(f"检测到{df['is_outlier'].sum()}个多变量异常点")
        outliers_indices = df.index[outliers == -1]
        result_df = df.loc[outliers_indices]
        # result_df.to_csv("extreme_features_isoforest.csv")

        self.history.append("检测数值列异常值(iso_forest)")
        return self


"""=============================================== 抽样数据 ============================================="""

"""系统抽样（等间隔抽样）"""


def systematic_resample(self, start_index: int = 5, step: int = 6) -> 'DataPreprocessor':
    """系统抽样（等间隔抽样）"""
    print("系统抽样（等间隔抽样）...")

    # 保证时间数据是排好序的
    original_shape = self.origin_df.shape
    self.origin_df = self.origin_df.iloc[start_index::step]
    resampled_shape = self.origin_df.shape

    print(f"等间隔抽样: 从索引 {start_index} 开始，步长 {step}，共 {len(self.origin_df)} 个样本")
    print(f"原始数据形状：{original_shape}")
    print(f"重采样后数据形状：{resampled_shape}")
    print(f"移除了 {original_shape[0] - resampled_shape[0]} 行")
    self.history.append("系统抽样(等间隔抽样)")
    return self  # 返回实例本身以支持链式调用


"""基于时间重采样"""


def time_based_resample(self, time_column: str = None,
                        freq: str = 'H',  # 重采样频率 ('H'-小时, 'D'-天, 'W'-周等)
                        aggregation: str = 'mean'  # 聚合方法 ('mean', 'sum', 'max', 'min', 'first', 'last')
                        ) -> 'DataPreprocessor':
    """适用于时间序列数据"""
    print("基于时间重采样...")

    if time_column not in self.origin_df.columns:
        print(f"时间列 '{time_column}' 不存在于数据中，未完成基于时间序列的采样")
        return self

    else:
        original_shape = self.origin_df.shape
        self.origin_df = (
            self.origin_df
            .set_index(time_column)
            .resample(freq)
            .agg(aggregation)
            .reset_index()
        )
        resampled_shape = self.origin_df.shape

        print(f"时间重采样: 频率 {freq}，聚合方法 {aggregation}")
        print(f"原始数据形状：{original_shape}")
        print(f"重采样后数据形状：{resampled_shape}")
        print(f"移除了 {original_shape[0] - resampled_shape[0]} 行")
        self.history.append("基于时间重采样")
        return self


"""=============================================== 极端数据 ============================================="""

"""处理异常值"""


def remove_outliers(self, method: Dict = None, target_col: str = None) -> 'DataPreprocessor':
    """业务判断：物理上不可能的值 / 极端数据，返回表格，根据索引处理
       目前支持简单物理异常 + 单列异常
       params: method
       {'name':'custom', } 自定义
       {'name':'singe_zscore','thredhold': 3 }
       {'name':'singe_iqr','thredhold': 1.5 }
       custom 方式可以不用提供target_cols，目前是写死的
       """

    if method is None:
        print("未提供异常值处理方式，暂不处理")
        return self

    if method['name'] == 'custom':
        print("处理异常值-物理不可能...")
        # 'wv'平均风速,'max. wv'最大风速列小于0，需将-9999 替换 为0，非删
        print(self.origin_df[self.origin_df['wv'] < 0]['wv'])
        print(self.origin_df[self.origin_df['max. wv'] < 0]['max. wv'])

        self.origin_df.loc[self.origin_df['wv'] == -9999.0, 'wv'] = 0
        self.origin_df.loc[self.origin_df['max. wv'] == -9999.0, 'max. wv'] = 0

        self.history.append("处理异常值-物理不可能")
        return self

    if method['name'] == 'singe_zscore':
        # 使用Z-score方法移除[单列]异常值，不for循环防止造成数据偏差
        print(f"处理{target_col}列异常值(zscore)...")

        # 防止有无用列被删除，可以更新一下 self.numeric_columns
        df = self.origin_df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if target_col in self.numeric_columns:
            cleaned_series = df[target_col].dropna()

            z_scores = np.abs(stats.zscore(cleaned_series))
            before_count = len(cleaned_series)  # 从dropna()后算，仅算异常值数
            normal_mask = (z_scores < method['threshold'])
            normal_indices = cleaned_series.index[normal_mask]
            self.origin_df = df.loc[normal_indices]
            after_count = len(self.origin_df)

            if before_count != after_count:
                print(f"列'{target_col}':移除了 {before_count - after_count} 个异常值")
            else:
                print(f"列'{target_col}':无异常值可移除")
        else:
            print(f"列'{target_col}'不在数据中")

        self.history.append(f"处理列异常值-z_scores")
        return self

    if method['name'] == 'singe_iqr':
        print(f"处理{target_col}列异常值(iqr)...")
        # 使用IQR方法移除异常值
        df = self.origin_df.copy()
        self.numeric_columns = df.select_dtype(include=[np.number]).columns.tolist()

        if target_col in self.numeric_columns:
            cleaned_series = df.dropna()

            if len(cleaned_series) >= 4 and len(cleaned_series.unique()) >= 4:
                Q1, Q3 = cleaned_series.quantile([0.25, 0.75])
                IQR = Q3 - Q1
                lower_bound = Q1 - method['threshold'] * IQR
                upper_bound = Q3 + method['threshold'] * IQR
                normal_mask = (cleaned_series >= lower_bound) & (cleaned_series <= upper_bound)

                before_count = len(cleaned_series)
                self.origin_df = df.loc[normal_mask]
                after_count = len(self.origin_df)

                if before_count != after_count:
                    print(f"列'{target_col}':移除了 {before_count - after_count} 个异常值")
                else:
                    print(f"列'{target_col}':无异常值可移除")
            else:
                print(f"列'{target_col}':唯一值数量小于4，iqr计算不可靠")
        else:
            print(f"列'{target_col}'不在数据中")

        self.history.append(f"处理列异常值-iqr")

        return self


"""=============================================== 其他特殊处理 ============================================="""

"""时间序列特别处理"""


def handle_time_col(self, col: str = None, format: str = None) -> 'DataPreprocessor':
    """处理时间列数据
       params: format 时间列原格式 """
    print(f"处理时间{col}数据...")
    df = self.origin_df.copy()

    # a.str -> datatime（原df中删除'Data Time'列）
    date_time = pd.to_datetime(df.pop(col), format=format)
    print(date_time.head(5))

    # b.将data_time中数据转换为时间戳格式的数据
    print(datetime.datetime.timestamp(date_time[5]))
    timestamp_s = date_time.map(datetime.datetime.timestamp)
    print(timestamp_s)

    # c.将时刻序列映射为正弦曲线序列
    day = 24 * 60 * 60  # 一天多少秒
    year = (365.2425) * day  # 一年多少秒

    df['Day sin'] = np.sin((timestamp_s * 2 * np.pi) / day)
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin((timestamp_s / year) * 2 * np.pi)
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    print("新增4列：['Day sin', 'Day cos', 'Year sin', 'Year cos']")

    self.origin_df = df
    print(f"时间列数据已处理，新增4列:'Day sin', 'Day cos', 'Year sin', 'Year cos'")
    print(self.origin_df.loc[0:2, ['Day sin', 'Day cos', 'Year sin', 'Year cos']])

    # d.将转换结果可视化
    viz = Visualization()
    viz.plot_time_signals(X=np.array(self.origin_df['Day sin'])[:25],  # 24小时
                          y=np.array(self.origin_df['Day cos'])[:25],
                          xlabel='时间[单位：时]（Time [h]）',
                          title='一天中的时间信号（Time of day signal）')

    self.history.append("处理时间数据-正余弦")
    return self


def handle_vec_col(self, dir_cols: List[str] = None, var_cols: List[str] = None) -> 'DataPreprocessor':
    """将'风向角度制'和'风速列极坐标'数据转换为风矢量
    dir_cols: 角度值的方向数据，
    var_cols: 极坐标的风速数据"""

    # 处理前:用极坐标（风速m/s）和风向（0-360）来描述风的强度和方向，
    # 处理后:用正交坐标系的两个维度（x轴和y轴）上风的强度，来描述上述'风速'和'风向' ['Wx', 'Wy', 'max Wx', 'max Wy']
    print("处理风矢量...")
    if dir_cols is None or var_cols is None:
        print("无'方向'(弧度制)数据、无'速度变量'数据需要处理")
        return self

    else:
        df = self.origin_df.copy()
        print(df.loc[0:2, dir_cols + var_cols])  # 平均风速、最大风速、风向（角度制）

        # 处理步骤：
        # a.将风向和风速列数据转换为风矢量，重新存入原数据框中
        # b.2D直方图--通过可视化的方式解释风矢量类型的数据由于原表风速和风向数据的原因

        # 原表风速和风向数据
        Visualization.plot_hist2d(x=df[dir_cols[0]],  # 'wd'
                                  y=df[var_cols[0]],  # 'wv'
                                  xlabel=f'{dir_cols[0]} 风向 [单位：度]',
                                  ylabel=f'{var_cols[0]} 风速 [单位：米/秒]')

        # 风矢量类型的数据
        wd_rad = df.pop(dir_cols[0]) * np.pi / 180  # 风向由角度制转换为弧度制
        wv = df.pop(var_cols[0])  # 先抓出 再丢了 将df中的wv列保存到wv中，并从原来的df中删除
        max_wv = df.pop(var_cols[1])

        df['Wx'] = wv * np.cos(wd_rad)  # 计算平均风力wv的x和y分量，保存到df的'Wx'列和'Wy'列中
        df['Wy'] = wv * np.sin(wd_rad)

        df['max Wx'] = max_wv * np.cos(wd_rad)  # 计算最大风力'max. mv'的x和y分量，保存到df的'max Wx'列和'max Wy'列中
        df['max Wy'] = max_wv * np.sin(wd_rad)

        # 不需要初始化任何东西，最适合静态方法，然后类名调用
        Visualization.plot_hist2d(x=df['Wx'],
                                  y=df['Wy'],
                                  xlabel='风的X分量[单位：m/s]',
                                  ylabel='风的Y分量[单位：m/s]')

        Visualization.plot_hist2d(x=df['max Wx'],
                                  y=df['max Wy'],
                                  xlabel='最大风的X分量[单位：m/s]',
                                  ylabel='最大风的Y分量[单位：m/s]')

        # 对比两图，分解后有利于我们观察风的状况：找到原点（0，0），
        # 假设向上为北，那么南方向的 风出现次数较多，此外我们还可以观察到东北-西南方向的风
        self.history.append("处理风矢量")
        return self


"""=============================================== 切分数据集并标准化 ============================================="""

"""切分数据集"""


def train_val_test_split(self,
                         train_size: float = 0.7,
                         val_size: float = 0.2,
                         test_size: float = 0.1) -> 'DataPreprocessor':
    """时间序列的分层，按顺序整体切3部分"""
    print("切分数据集...")
    df = self.origin_df.copy()

    n = len(df)
    self.trainSets = df.iloc[0:int(n * train_size)]
    self.valSets = df.iloc[int(n * train_size):int(n * (train_size + val_size))]
    self.testSets = df.iloc[int(n * (train_size + val_size)):]

    print(
        f"数据分割完成: 训练集 {len(self.trainSets)} 样本, 验证集 {len(self.valSets)} 样本,测试集 {len(self.testSets)} 样本")
    self.history.append("切分数据集")
    return self


"""检查验证集"""


def has_validation_set(self) -> bool:
    """检查是否有验证集"""
    return not self.valSets.empty and len(self.valSets) > 0


"""数据标准化/归一化"""


def unify_feature_scaling(self, transformers: List = None) -> 'DataPreprocessor':  # 即zscore（原值-均值）/ 标准差
    """统一标准化和归一化数值特征
    'minmax', {'feature_range': (0, 1), 'threshold': 1.5, 'columns': ['T']}
             归一化 (Normalization)	(x - min) / (max - min) scaler = MinMaxScaler(feature_range=(0, 1))  # 可选-1，1
    'std_scaler':{'threshold':3,'columns':[]},
             标准化 zscore 3倍标准差 (x-mean)/std
    'robust_scaler': {'quantile_range':(25, 75), 'columns':[]}
            鲁棒 X_scaled = (X - median) / IQR
    """
    if transformers is None:
        print("统一数据格式但未提供方法")
        return self

    self.scalers = {}  # 保存每个列独立的scaler
    self.constant_values = {}  # 保存常数列的原始值
    for method, config in transformers:
        if method == 'minmax':
            print("数据归一化(min_max)...")
            # Min-Max 归一化（使用训练集统计量）

            for col in config['columns']:
                if self.trainSets[col].notna().sum() <= 1:  # 数据要求至少2个
                    print(f"列{col}:训练集数据不足，跳过归一化")
                    self.scalers[col] = None

                else:
                    train_col = self.trainSets[col].dropna()
                    col_max = train_col.max()
                    col_min = train_col.min()

                    if col_max == col_min:  # 所有值相同的情况
                        self.trainSets[col] = 0.5
                        self.constant_values[col] = col_max

                        if self.has_validation_set() and col in self.valSets.columns:
                            self.valSets[col] = 0.5
                        if not self.testSets.empty and col in self.testSets.columns:
                            self.testSets[col] = 0.5
                        print(f"列 '{col}': 最大值等于最小值，设为0.5")
                        # 标记这个列不需要scaler
                        self.scalers[col] = 'constant'

                    else:
                        # 定义： 只在训练集上拟合（计算参数）
                        scaler = MinMaxScaler(feature_range=config['feature_range'])  # 可选-1，1
                        scaler.fit(train_col)
                        # 将拟合好的scaler存入字典，以列名为键
                        self.scalers[col] = scaler

                        # 注意 1.transform要求的输入格式，2.转换完赋回去
                        self.trainSets[col] = scaler.transform(self.trainSets[col].values.reshape(-1, 1)).flatten()
                        print(
                            f"训练集列{col}:min_max 归一化完成")  # 1列Series(n,)->2D array(n,)。转换完用flatten()  压回1维(n,)
                        if self.has_validation_set() and col in self.valSets.columns:
                            self.valSets[col] = scaler.transform(self.valSets[col].values.reshape(-1, 1)).flatten()
                            print(f"验证集列{col}:min_max 归一化完成")
                        if not self.testSets.empty and col in self.testSets.columns:
                            self.testSets[col] = scaler.transform(
                                self.testSets[col].values.reshape(-1, 1)).flatten()
                            print(f"测试集列{col}:min_max 归一化完成")

            self.history.append("归一化min_max完成")

        if method == 'std_scaler':
            # zscore: （value-均值）/ 标准差 3 (dropna)
            print("数据标准化(zscore)...")

            for col in config['columns']:
                if self.trainSets[col].notna().sum() <= 1:  # 数据不足,至少2个非空
                    print(f"列{col}:训练集数据不足，跳过标准化")
                    self.scalers[col] = None

                else:
                    train_col = self.trainSets[col].dropna()  # 自动处理
                    col_mean = train_col.mean()  # 防止数据泄漏 只用训练集的均值和标准差
                    col_std = train_col.std()

                    if col_std == 0:  # 处理零标准差情况
                        self.constant_values[col] = col_mean
                        print(f"列{col} 的均值为{col_mean}，标准差为0 ，设为常数0")
                        self.trainSets[col] = 0  # 所有值相同，设为0
                        if self.has_validation_set() and col in self.valSets.columns:
                            self.valSets[col] = 0
                        if not self.testSets.empty and col in self.testSets.columns:
                            self.testSets[col] = 0
                        self.scalers[col] = 'constant'

                    else:  # std_z > 1e-8 避免除零错误
                        scaler = StandardScaler()
                        scaler.fit(train_col)
                        self.scalers[col] = scaler

                        self.trainSets[col] = scaler.transform(self.trainSets[col].values.reshape(-1, 1)).flatten()
                        print(f"训练集列{col}:z_score 标准化完成")
                        if self.has_validation_set() and col in self.valSets.columns:
                            self.valSets[col] = scaler.transform(self.valSets[col].values.reshape(-1, 1)).flatten()
                            print(f"验证集列{col}:z_score 标准化完成")
                        if not self.testSets.empty and col in self.testSets.columns:
                            self.testSets[col] = scaler.transform(
                                self.testSets[col].values.reshape(-1, 1)).flatten()
                            print(f"测试集列{col}:z_score 标准化完成")
                        print(f"列 '{col}': 标准化完成 (均值: {col_mean:.2f}, 标准差: {col_std:.2f})")

            self.history.append("数值列标准化(zscore)")

        if method == 'robust_scaler':
            print("数据鲁棒标准化(robust)...")

            for col in config['columns']:
                train_non_null = self.trainSets[col].dropna()
                n_samples = len(train_non_null)

                # 样本数 ≤ 1
                if n_samples <= 1:  # 数据要求至少2个
                    print(f"列{col}:训练集数据不足（{n_samples}个样本），跳过标准化")
                    self.scalers[col] = None
                    continue

                # 样本数 2-4，四分位数计算不稳定(分为:值相同 + 值不同的情况)
                elif n_samples <= 4:  # 处理：使用 中位数和标准差 替代 中位数和IQR 公式：(X - median) / std
                    print(f"列{col}:样本数较少({n_samples}个)，使用中位数和标准差代替 IQR")
                    col_median = train_non_null.median()
                    col_std = train_non_null.std()

                    if col_std == 0:  # 所有值相同的情况:标准差为0，设为0，记录中位数
                        self.trainSets[col] = 0
                        self.constant_values[col] = col_median

                        if self.has_validation_set() and col in self.valSets.columns:
                            self.valSets[col] = 0
                        if not self.testSets.empty and col in self.testSets.columns:
                            self.testSets[col] = 0
                        print(f"列 '{col}': 标准差为0，设为0")
                        self.scalers[col] = 'constant'
                    else:  # 值不相同的情况，手动实现基于标准差的鲁棒缩放
                        self.trainSets[col] = (self.trainSets[col] - col_median) / col_std
                        print(f"训练集{col}：使用中位数和标准差进行鲁棒标准化")

                        # 保存scaler信息供验证集和测试集使用
                        self.scalers[col] = {
                            'type': 'manual_robust',
                            'median': col_median,
                            'scale': col_std
                        }
                        if self.has_validation_set() and col in self.valSets.columns:
                            self.valSets[col] = (self.valSets[col] - col_median) / col_std
                            print(f"验证集列{col}：使用中位数和标准差进行鲁棒标准化")
                        if not self.testSets.empty and col in self.testSets.columns:
                            self.testSets[col] = (self.testSets[col] - col_median) / col_std
                            print(f"测试集列{col}：使用中位数和标准差进行鲁棒标准化")

                else:  # 数据足够（有一个特殊IQR=0）
                    q1 = train_non_null.quantile(0.25)
                    q3 = train_non_null.quantile(0.75)
                    iqr = q3 - q1

                    if iqr == 0:  # 原公式分母为0
                        self.trainSets[col] = 0
                        self.constant_values[col] = train_non_null.median()
                        if self.has_validation_set() and col in self.valSets.columns:
                            self.valSets[col] = 0
                        if not self.testSets.empty and col in self.testSets.columns:
                            self.testSets[col] = 0
                        print(f"列 '{col}': IQR为0，设为0")
                        self.scalers[col] = 'constant'
                    else:
                        scaler = RobustScaler(
                            with_centering=True,  # 减去中位数
                            with_scaling=True,  # 除以IQR
                            quantile_range=(25, 75)  # scikit-learn 使用百分比形式
                        )
                        scaler.fit(train_non_null.values.reshape(-1, 1))
                        self.scalers[col] = scaler
                        self.trainSets[col] = scaler.transform(self.trainSets[col].values.reshape(-1, 1)).flatten()
                        print(
                            f"训练集列{col}: robust标准化完成 (median={scaler.center_[0]:.3f}, IQR={1 / scaler.scale_[0]:.3f})")
                        # 转换验证集和测试集
                        if self.has_validation_set() and col in self.valSets.columns:
                            self.valSets[col] = scaler.transform(
                                self.valSets[col].values.reshape(-1, 1)
                            ).flatten()
                            print(f"验证集列{col}: robust标准化完成")

                        if not self.testSets.empty and col in self.testSets.columns:
                            self.testSets[col] = scaler.transform(
                                self.testSets[col].values.reshape(-1, 1)
                            ).flatten()
                            print(f"测试集列{col}: robust标准化完成")

            self.history.append("鲁棒标准化robust完成")


def get_history(self):
    return self.history


def get_data(self):
    return self.origin_df


def get_train_val_test_data(self):
    if not self.trainSets.empty:
        return self.trainSets.copy(), self.valSets.copy(), self.testSets.copy()
    else:
        print("切分数据集操作尚未执行")
        return None


def save_scalers(self, filename: str = 'scalers.pkl'):
    filepath = os.path.join(self.dir_path, filename)
    joblib.dump(self.scalers, filepath)
    print(f"Scalers saved to: {filepath}")
    return filepath


def save_constant_values(self, filename: str = 'constant.pkl'):
    filepath = os.path.join(self.dir_path, filename)
    joblib.dump(self.constant_values, filepath)
    print(f"constant_values saved to: {filepath}")
    return filepath
