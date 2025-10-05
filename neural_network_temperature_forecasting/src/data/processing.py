from sklearn.base import BaseEstimator, TransformerMixin
from functools import wraps
from sklearn.utils.validation import check_array, check_consistent_length
from pathlib import Path
import codecs
import os
import csv
import glob
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from scipy import stats
import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from data.exploration import Visualization
import joblib
import matplotlib

matplotlib.interactive(False)  # 明确关闭交互模式


def validate_input(validate_y=True, allow_empty=False, **param_checks):
    """输入验证装饰器
        Args:
            validate_y: 是否验证y参数
            allow_empty: 是否允许空数据集（用于某些特殊场景）
    """

    def decorator(method):
        @wraps(method)  # 保持原始函数名和文档字符串
        def wrapper(self, X, y=None, *args, **kwargs):
            # 基础X验证
            if X is None:
                raise ValueError("输入数据X不能为None")

            if not allow_empty:
                if hasattr(X, 'shape'):
                    if X.shape[0] == 0:
                        raise ValueError("输入数据X不能为空")
                    if len(X.shape) > 1 and X.shape[1] == 0:
                        raise ValueError("输入数据X的特征列不能为空")
                elif hasattr(X, '__len__') and len(X) == 0:
                    raise ValueError("输入数据X不能为空")

            if not isinstance(X, (pd.DataFrame, np.ndarray, list)):
                try:
                    X = check_array(X, ensure_2d=False)
                except:
                    raise TypeError(f"输入数据X必须是DataFrame,array或者list格式，其为{type(X)}")

            # y的验证
            if validate_y and y is not None:
                if not isinstance(y, (pd.Series, np.ndarray, list)):
                    try:
                        y = check_array(y, ensure_2d=False)
                    except:
                        raise TypeError(f"数据数据y必须是Series，array或者list，其为{type(y)}")

                # 检查样本数量一致性
                x_len = X.shape[0] if hasattr(X, 'shape') else len(X)
                y_len = y.shape[0] if hasattr(y, 'shape') else len(y)

                if x_len != y_len:
                    raise ValueError(f"X和y长度不一致：{x_len} vs {y_len}")

            # 检查参数
            for param_name, check_func in param_checks.items():
                if hasattr(self, param_name):
                    value = getattr(self, param_name)
                    if not check_func(value):
                        raise ValueError(f"无效参数值 {param_name}:{value}")

            return method(self, X, y, *args, **kwargs)

        return wrapper

    return decorator


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

    @validate_input(validate_y=False)
    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self._describe_data(df)

        return self

    @validate_input(validate_y=True)
    def transform(self, X, y=None):
        if y is not None:
            return X, y
        return X

    def _describe_data(self, df: pd.DataFrame):
        self.stats = df.describe()
        print("描述性分析如下：")
        print(self.stats)

        print("\n 数据类型：")
        print(df.dtypes)

        print("\n 缺失值统计：")
        print(df.isna().sum())


"""删除无用列"""


class DeleteUselessCols(BaseEstimator, TransformerMixin):
    def __init__(self, target_cols: Optional[list] = None):
        self.target_cols = target_cols or []

    @validate_input(validate_y=False)  # 删除列不需要验证y
    def fit(self, X, y=None):
        if self.target_cols is None:
            print("调用删除无用列功能，但未填写列名")

        # 检查目标列是否存在
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        if self.target_cols:
            existing_cols = [col for col in self.target_cols if col in df.columns]
            missing_cols = [col for col in self.target_cols if col not in df.columns]
            if missing_cols:
                print(f"警告: 以下列不存在，将被忽略: {missing_cols}")
            self.target_cols = existing_cols
        else:
            self.target_cols = []

        return self

    @validate_input(validate_y=True)  # transform时需要验证y
    def transform(self, X, y=None):
        print("移除无用列...")
        if not self.target_cols:
            print("没有需要删除的列")
            return X

        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        X_cleaned = X.drop(columns=self.target_cols, axis=1)
        print(f"移除了{len(self.target_cols)}个列: {self.target_cols}")

        if y is not None:
            return X_cleaned, y  # y保持不变
        return X_cleaned


"""移除重复值"""


class RemoveDuplicates(BaseEstimator, TransformerMixin):
    def __init__(self, download_config: Optional[Dict[str, Union[str, bool]]] = None):
        self._has_downloaded = False  # 防止重复下载
        if download_config is None:
            self.download_config = {
                'enabled': False,
                'path': './output',
                'filename': 'duplicate_rows.csv'}
        else:
            self.download_config = dict(download_config)  # 确保是字典，否则get黄色

    @validate_input(validate_y=True)
    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X  # fit 不用管y 还可以叫df

        self.retain_indices_ = ~df.duplicated(keep='first')
        return self

    @validate_input(validate_y=True)
    def transform(self, X, y=None):
        print("移除重复行...")
        if ~self.retain_indices_ is None or len(~self.retain_indices_) == 0:
            print("无重复值需要处理")
            return X

        X_ = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        initial_count = len(X_)
        X_cleaned = X_[self.retain_indices_]
        removed_count = initial_count - len(X_cleaned)
        print(f"移除了{removed_count}个重复行")

        if hasattr(self, 'download_config') and self.download_config.get('enabled', False):
            self._download_duplicate_rows(X_)

        if y is not None:
            if hasattr(y, 'iloc'):
                y_cleaned = y.iloc[self.retain_indices_]
            else:
                y_cleaned = y[self.retain_indices_]

            return X_cleaned, y_cleaned

        return X_cleaned

    def _download_duplicate_rows(self, df):
        if self._has_downloaded:
            print("重复数据已下载过，跳过本次下载")
            return
        if ~self.retain_indices_ is None or len(~self.retain_indices_) == 0:
            print("无重复数据可下载")
            return

        print("下载重复数据(包括唯一行及重复行...")
        # 所有重复的行都为True，只有唯一的行为False。默认'first'重复值中的第一个，是False被保留
        duplicate_mask = df.duplicated(keep=False)
        duplicate_rows = df[duplicate_mask]  # 注意这里和retain_indices不一致 方便检查 保留唯一行
        duplicate_count = len(duplicate_rows)
        print(f"识别到 {duplicate_count} 行(包含唯一行和其所有重复行")

        path = self.download_config.get('path', './output')
        filename = self.download_config.get('filename', 'duplicate_rows')

        if not filename.endswith('.csv'):
            filename += '.csv'
        if path and duplicate_rows is not None and len(duplicate_rows) > 0:
            # 确保目录存在
            os.makedirs(path, exist_ok=True)
            # 构建完整文件路径
            file_path = os.path.join(path, filename)

            # 下载重复数据
            try:
                duplicate_rows.to_csv(file_path, index=False)
                print(f"重复数据已下载到: {file_path}")
                self._has_downloaded = True
            except Exception as e:
                print(f"下载重复数据失败: {e}")

        else:
            print("未配置下载路径或无重复数据可下载")


"""一般问题列正则处理"""


class ProblemColumnsFixed(BaseEstimator, TransformerMixin):

    def __init__(self, problem_columns: Optional[list] = None):
        self.problem_columns = problem_columns or []
        self.columns_to_process_ = []

    @validate_input(validate_y=False)
    def fit(self, X, y=None):
        if self.problem_columns is None:
            print("使用修复列功能，但未指定待修复问题列")
            return self

        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        # 确认问题列是否存在
        self.columns_to_process_ = [col for col in self.problem_columns if col in df.columns]
        missing_cols = [col for col in self.problem_columns if col not in df.columns]

        if missing_cols:
            print(f"以下列不存在，将跳过: {missing_cols}")

        print(f"将处理 {len(self.columns_to_process_)} 个问题列: {self.columns_to_process_}")
        return self

    @validate_input(validate_y=True)
    def transform(self, X, y=None):
        """应用正则清洗转换"""
        if not self.columns_to_process_:
            return X
        print("一般问题列正则处理...")
        X_ = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        processed_count = 0

        for col in self.columns_to_process_:
            if col in X_.columns:
                # 记录处理前信息
                non_null_count: int = X_[col].notna().sum()
                if non_null_count > 0:
                    sample_value = X_[col].iloc[0]
                    print(f"问题列第一个元素：{sample_value}")

                    X_[col] = (X_[col]
                    .astype(str)
                    .str.extract(r'([-+]?\d*\.?\d+)', expand=False)[0])  # expand=False 返回Series

                    X_[col] = pd.to_numeric(X_[col], errors='coerce')
                    processed_count += 1
                    print(f"列 '{col}':已转文本，正则清洗，转回数值")
        print(f"正则清洗完成: 成功处理 {processed_count}/{len(self.columns_to_process_)} 个列")

        if y is not None:
            return X_, y

        return X_


"""修复问题列-列包含df"""


class SpecialColumnsFixed(BaseEstimator, TransformerMixin):
    def __init__(self, problem_columns: Optional[list] = None):
        self.problem_columns = problem_columns or []
        self.columns_to_process_ = []

    @validate_input(validate_y=False)
    def fit(self, X, y=None):

        if self.problem_columns is None:
            print("使用'特别修复'列功能，但未指定待修复问题列")
            return self

        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

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

    @validate_input(validate_y=True)
    def transform(self, X, y=None):
        print("开始修改问题列...")
        if not self.columns_to_process_:
            print("没有需要修复的列")
            return X

        X_ = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        for col in self.columns_to_process_:  # fit判断过，但可能单独调transform
            if col not in X_.columns:
                print(f"列 '{col}' 不存在于数据中，跳过")
                continue

            print(f"\n修复列: {col}")
            series = X_[col].copy()
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
            X_[col] = series_fixed

            # 验证修复结果
            if len(X_[col]) > 0:
                sample_value = X_[col].iloc[0]
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

        if y is not None:
            return X_, y
        return X_


"""识别列类型"""


class ColumnsTypeIdentify(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numeric_columns = None
        self.categorical_columns = None
        self.other_columns = None

    @validate_input(validate_y=False)
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

    @validate_input(validate_y=True)
    def transform(self, X, y=None):
        if y is not None:
            return X, y
        return X


"""时间类型工具"""


class TimeTypeConverter:
    """时间类型转换工具类"""

    def detect_time_type(self, series):
        """检测时间序列的类型"""
        if len(series) == 0:
            return "empty"

        clean_series = series.dropna()
        sample = clean_series.iloc[0] if len(clean_series) > 0 else None

        # 检查datetime64类型
        if pd.api.types.is_datetime64_any_dtype(clean_series):
            return "datetime64"

        # 检查Timestamp类型
        if isinstance(sample, pd.Timestamp):
            return "timestamp"

        # 检查字符串
        if isinstance(sample, str):
            return "string"

        # 检查数值类型
        if pd.api.types.is_numeric_dtype(clean_series):
            if sample is None:
                return "unknown_numeric"
            if len(clean_series) < 5:
                return "unknown_numeric"

            # 判断数值型是否是Excel日期还是Unix时间戳？
            return self._detect_numeric_time(series=clean_series)

        return "unknown"

    def convert_to_datetime(self, series, time_type=None, format=None):
        """根据类型转换为datetime"""
        if time_type is None:
            time_type = self.detect_time_type(series)

        print(f"检测到类型: {time_type}")

        if time_type == "datetime64":
            print("已经是datetime64类型，无需转换")
            return series

        elif time_type == "timestamp":
            print("Timestamp对象序列，转换为datetime64...")
            return pd.to_datetime(series)

        elif time_type == "string":
            print("字符串格式，解析为datetime...")
            if format is not None:
                return pd.to_datetime(series, format=format, errors='coerce')
            else:
                return pd.to_datetime(series, errors='coerce')

        elif time_type == "unix_timestamp":
            print("Unix时间戳，根据数值范围判断单位")
            # 根据数值大小判断单位
            if series.max() < 10000000000:  # 约到2286年
                unit = 's'  # 秒
            elif series.max() < 10000000000000:  # 约到2286年
                unit = 'ms'  # 毫秒
            else:
                unit = 'us'  # 微秒
            print(f"使用单位: {unit}")
            return pd.to_datetime(series, unit=unit)

        elif time_type == "excel_date":
            print("Excel日期数字，使用origin转换")
            # Excel的日期系统从1900-01-01开始(已知闰年错误)pandas修复起始时间1899-12-30
            return pd.to_datetime(series, unit='D', origin='1899-12-30')

        else:
            print("未知类型，尝试自动解析")
            return pd.to_datetime(series, errors='coerce')

    def get_conversion_info(self, series, time_type=None, format=None):
        """获取转换的详细信息"""
        if time_type is None:
            time_type = self.detect_time_type(series)

        datetime_series = self.convert_to_datetime(series, time_type, format)

        info = {
            'original_type': time_type,
            'converted_type': str(datetime_series.dtype),
            'success_rate': (1 - datetime_series.isna().mean()) * 100,
            'na_count': datetime_series.isna().sum(),
            'sample_values': {
                'original': series.head(3).tolist(),
                'converted': datetime_series.head(3).tolist()
            }
        }

        return info, datetime_series

    def _detect_numeric_time(self, series):
        """检测用数值型时间列"""

        stats = {
            'min': series.min(),
            'max': series.max()
        }

        min_val, max_val = stats['min'], stats['max']
        unix_range = (1000000000, 4102444800)  # 2001-01-01 到 2100-01-01
        excel_range = (32874, 54788)  # 1990-01-01 到 2100-01-01

        if (unix_range[0] <= min_val <= max_val <= unix_range[1] and
                self._check_digit_pattern(series, [10, 13]) and
                self._is_monotonic_increasing(series)
                and self._comprehensive_cv_check(series)  # 等时间间隔 检查(众数、CV检查)
        ):
            return 'unix_timestamp'

        if (excel_range[0] <= min_val <= max_val <= excel_range[1] and
                self._check_digit_pattern(series, [5, 6]) and
                self._is_monotonic_increasing(series)
                and self._comprehensive_cv_check(series)  # 等时间间隔 检查(众数、CV检查)
        ):
            return 'excel_date'

    def _check_digit_pattern(self, series, range):
        """检查数字位数模式
        Unix 时间戳：通常是10位（秒）或13位（毫秒）Excel 日期：通常是5-6位整数"""
        str_sample = series.astype(int).astype(str)
        digit_counts = str_sample.str.len().value_counts()

        # 计算主要位数的占比
        if len(digit_counts) > 0:
            main_digit_count = digit_counts.index[0]
            main_digit_ratio = digit_counts.iloc[0] / len(series)  # 只有主要位数占比足够高时才认为有固定模式

            if main_digit_ratio > 0.8:
                return main_digit_count in range
        else:
            return False

    def _is_monotonic_increasing(self, series):
        if len(series) < 2:
            return False
        sample = series.head(100).sort_index()
        return sample.is_monotonic_increasing  # Pandas属性 数据太少可能无法判断。允许相等

    def _common_cv_pattern(self, series):
        """检查是否具有等时间间隔均匀采样模式(变异系数 相对离散程度)
            整体CV < 0.3，整数CV < 0.5"""
        sample = series.head(50)
        integer_parts = sample.astype(int)
        fractional_parts = sample % 1

        if len(sample) > 1:
            # 检查整体序列的CV
            diffs = sample.diff().dropna()
            overall_cv = diffs.std() / diffs.mean() if diffs.mean() != 0 else float('inf')

            # 检查整数部分的CV（应该较小，因为主要是日期）
            int_diffs = integer_parts.diff().dropna()
            int_cv = int_diffs.std() / int_diffs.mean() if int_diffs.mean() != 0 else float('inf')

            # 计算小数部分的cv(如果存在)
            fractional_parts = fractional_parts[fractional_parts > 0]  # 过滤掉0
            if len(fractional_parts) > 1:
                frac_diffs = fractional_parts.diff().dropna()
                frac_cv = frac_diffs.std() / frac_diffs.mean() if frac_diffs.mean() != 0 else float('inf')
            else:
                frac_cv = float('inf')  # 明确标识特殊情况，在函数后续的逻辑判断中被处理

            return {
                'overall_cv': overall_cv,
                'int_cv': int_cv,
                'frac_cv': frac_cv,
                'has_fractional': len(fractional_parts) > 0

            }

    def _comprehensive_cv_check(self, series):
        # 先排序
        sorted_series = series.sort_index()  # 时间不可逆 原始索引大多跟采集时间来

        if len(sorted_series.dropna()) < 5:
            return False

        checks = []

        # 检测间隔模式，忽略异常值
        checks.append(self._detect_interval_pattern(sorted_series))

        # 检测主要间隔模式 基于中位数
        checks.append(self._cv_pattern_median_based(sorted_series))

        checks.append(self._cv_pattern_chunked(sorted_series))

        pass_count = sum(checks)
        return pass_count >= 1

    def _detect_interval_pattern(self, series):
        """检测间隔模式，忽略异常值"""
        sample = series.head(100)
        if len(sample) <= 1:
            return False

        diffs = sample.diff().dropna()
        if len(diffs) < 2:
            return False

        # 使用众数（mode）
        from collections import Counter
        diff_counter = Counter(diffs.round(3))  # 四舍五入避免浮点误差

        # 找最常见间隔
        if diff_counter:
            most_common_diff, most_common_count = diff_counter.most_common(1)[0]  # most_common(1)前1个
            ratio = most_common_count / len(diffs)

            return ratio > 0.6  # 如果超过60%的间隔相同，认为是规则采样

        return False

    def _cv_pattern_median_based(self, series):
        sample = series.head(50)

        if len(sample) <= 1:
            return False
        diffs = sample.diff().dropna()

        if len(diffs) < 2:
            return False

        median_diff = diffs.median()
        if median_diff == 0:
            return False

        # 基于中位数的“变异系数”
        mad = (diffs - median_diff).abs().median()  # 中位数绝对偏差
        robust_cv = mad / median_diff

        return robust_cv < 0.5

    def _cv_pattern_chunked(self, series, chunk_size=10):
        sample = series.head(100).sort_index()

        if len(sample) <= chunk_size:
            return self._cv_pattern_median_based(sample)

        chunk_results = []

        step = max(1, chunk_size // 2)
        for i in range(0, len(sample), step):
            chunk = sample.iloc[i: min(i + chunk_size, len(sample))]

            # 只有块足够大时才检测
            if len(chunk) >= max(5, chunk_size // 2):
                chunk_cv_ok = self._cv_pattern_median_based(chunk)
                chunk_results.append(chunk_cv_ok)

        if chunk_results:
            pass_ratio = sum(chunk_results) / len(chunk_results)
            return pass_ratio > 0.7

        return False


"""时间序列特别处理"""


class ProcessTimeseriesColumns(BaseEstimator, TransformerMixin):
    def __init__(self, col: Optional[str] = None,
                 format: Optional[str] = None,
                 interactive: bool = True,
                 auto_detect_string_format: bool = False  # 是否自动检测格式format
                 ):

        self.time_column = col
        self.time_converter = TimeTypeConverter()  # 使用时间类型转换器
        self.detected_time_type = None  # 存储检测到的时间类型
        self.interactive = bool(interactive)  # 确保属性存在
        self.sample_data_ = None
        self.auto_detect_string_format = auto_detect_string_format
        self.format = format
        self.common_formats = None
        self.fitted_ = False

    @validate_input(validate_y=False)
    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        # 存储必要的元数据
        self.sample_data_ = df.head(100).copy()

        if self.time_column is None:  # None []
            print("未指定待处理时间列，检查原数据的全部可能列...")

            potential_time_cols = []
            for col in self.sample_data_.columns:
                col_type = self.time_converter.detect_time_type(self.sample_data_[col])

                # 自动检测时，排除string/整数型时间列，仅支持指定
                if col_type in ['unix_timestamp', 'excel_date', 'timestamp', 'datetime64']:
                    potential_time_cols.append((col, col_type))
                    print(f"_列'{col}':类型 {col_type}")

                elif col_type == 'string':

                    sample_str = str(df[col].iloc[0]) if len(df[col]) > 0 else ""
                    date_indicators = ['-', '/', ':', '202', '201', '200', '199']

                    time_keywords = ['time', 'date', 'day', 'year', 'month', 'hour', 'minute', 'second']
                    col_name_lower = col.lower()

                    has_date_content = any(indicator in sample_str for indicator in date_indicators)
                    has_time_keyword = any(keyword in col_name_lower for keyword in time_keywords)

                    if has_date_content and has_time_keyword:
                        potential_time_cols.append((col, 'string(time-like)'))
                        print(f"_列 '{col}':字符串但包含时间格式和关键词")

            if potential_time_cols:
                # 检查是否需要交互式选择
                should_use_interactive = (len(potential_time_cols) > 1 and
                                          getattr(self, 'interactive', False))  # false：若没有属性，默认返回。初始化已保证有交互属性

                if should_use_interactive:
                    print("启用交互式时间列选择模式")
                    self.time_column = self._interactive_select_time_column(potential_time_cols)
                    for col, col_type in potential_time_cols:
                        if col == self.time_column:
                            self.detected_time_type = col_type
                    print(f"时间列 '{self.time_column}'，类型: {self.detected_time_type}")

                else:
                    # 非交互模式下自动选择第一个
                    self.time_column = potential_time_cols[0][0]  # 有点随机，数据存在多个时间列
                    self.detected_time_type = potential_time_cols[0][1]
                    print(
                        f"取第一个元组的结果，识别到可能的时间列 '{self.time_column}'，类型: {self.detected_time_type}")

                # 尝试检测字符串的format
                if self.detected_time_type == 'string':
                    if self.auto_detect_string_format and hasattr(self, 'sample_date_'):
                        if self.sample_data_ is not None and self.time_column in self.sample_data_.columns:
                            self._smart_format_processing(self.sample_data_[self.time_column])

            else:
                print("未找到可识别的时间列")
                self.time_column = None
        else:
            # 检查'指定的列'是否实际存在
            if self.time_column not in self.sample_data_.columns:
                print(f"警告: 该指定时间列不存在: {self.time_column}")
                self.time_column = None
            else:  # 检测‘指定列’的时间类型
                self.detected_time_type = self.time_converter.detect_time_type(self.sample_data_[self.time_column])
                print(f"检测到时间列{self.time_column}，时间类型：{self.detected_time_type}")

                if self.detected_time_type == 'string':

                    if self.format and self.auto_detect_string_format == False:
                        print(f"提供的字符串类型时间格式：{self.format}")
                    elif self.format and self.auto_detect_string_format:
                        print(f"已提供的字符串类型时间格式：{self.format}，将不启动 auto_detect_string_format 自动检测")
                    elif self.format is None and self.auto_detect_string_format == False:
                        print(
                            f"未提供的字符串类型时间格式 format ，且未使用 auto_detect_string_format 自动检测，pandas将自动推断")
                    else:
                        print(f"未提供的字符串类型时间格式 format，将使用自动检测...")
                        self._smart_format_processing(self.sample_data_[self.time_column])

        self.fitted_ = True
        return self

    @validate_input(validate_y=True)
    def transform(self, X, y=None):
        print("处理时间序列数据...")
        if not self.fitted_:
            raise ValueError("必须先调用fit方法")
        if self.time_column is None:
            print("无时间序列需处理")
            return X

        X_ = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        # 时间列可能包含：整数/浮点数、datetime对象、Timestamp对象、numpy.datetime64对象、混合数据

        # 使用TimeTypeConverter进行智能转换，如果是字符串类型时间，强制给出format参数
        print(f"使用TimeTypeConverter处理时间列‘{self.time_column}'...")

        conversion_info, datetime_series = self.time_converter.get_conversion_info(
            X_[self.time_column],
            self.detected_time_type,
            self.format
        )

        # 更新列数据
        X_[self.time_column] = datetime_series
        print(f"原始类型: {conversion_info['original_type']}")
        print(f"转换后类型: {conversion_info['converted_type']}")
        print(f"转换成功率: {conversion_info['success_rate']:.2f}%")
        print(f"NaN数量: {conversion_info['na_count']}")
        # 如果转换成功率太低，发出警告
        if conversion_info['success_rate'] < 80:
            print(f"警告: 时间列转换成功率较低 ({conversion_info['success_rate']:.2f}%)")

        # add 新增时间特征列
        self._new_features_from_timecols(df=X_, col=self.time_column)

        # add 周期编码时间列 （转换为Unix时间戳秒数,datetime64每个元素都是timestamp实例）
        self._cyclic_encoding(df=X_, col=self.time_column)

        try:
            viz = Visualization()  # 将转换结果可视化
            viz.plot_time_signals(X=np.array(X_['Day sin'])[:25],  # 24小时
                                  y=np.array(X_['Day cos'])[:25],
                                  xlabel='时间[单位：时]（Time [h]）',
                                  title='一天中的时间信号（Time of day signal）')
        except:
            print("可视化组件不可用，跳过绘图")

        if y is not None:
            return X_, y
        return X_

    def _interactive_select_time_column(self, potential_time_cols):
        """交互式选择字符串型时间列"""

        if not getattr(self, 'interactive', False):
            return None  # 交互属性是true 就继续往下

        print('\n' + '=' * 50)
        print("发现多个可能的时间列，请选择：")

        # 使用样本数据（如果有）
        sample_df = getattr(self, 'sample_data_', None)

        for i, (col, col_type) in enumerate(potential_time_cols):
            if sample_df is not None and col in sample_df.columns:
                sample_values = sample_df[col].dropna().head(5)
                sample_str = '|'.join([str(val) for val in sample_values])
                print(f"样例：{sample_str}")
                na_count = sample_df[col].isna().sum()
                total_count = len(sample_df[col])
                print(f" {i + 1}. {col} (类型：{col_type})，空值：{na_count} / {total_count})")

            else:
                print(f"{i + 1}.{col} (类型：{col_type})")

        while True:
            try:
                choice = input(f"请输入选择（1-{len(potential_time_cols)})，或输入'skip'跳过： ").strip()
                if choice.lower() == 'skip':
                    print("跳过时间列选择")
                    return None

                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(potential_time_cols):
                    selected_col = potential_time_cols[choice_idx][0]
                    selected_col_type = potential_time_cols[choice_idx][1]
                    print(f"已选择时间列：{selected_col}")

                    # 针对字符串时间类型进行的format智能处理 用sample先测试
                    if sample_df is not None and selected_col in sample_df.columns:
                        if selected_col_type == 'string(time-like)':
                            self._smart_format_processing(sample_df[selected_col])
                            return selected_col
                        else:
                            return selected_col

                else:
                    print(f"输入 1-{len(potential_time_cols)}之间的数字")

            except ValueError:
                print("请输入有效数字")
            except KeyboardInterrupt:
                print("\n用户中断选择")
                return None

    def _interactive_select_time_format(self, time_series):
        """交互式选择时间格式"""
        if len(time_series) == 0:
            return None

        samples = time_series.dropna().head(3)
        if len(samples) == 0:
            return None

        print(f"\n选择的时间列样本")
        for i, sample in enumerate(samples):
            print(f" {i + 1}. {sample}")

        print(f"\n请选择时间格式：")
        for i, (fmt, desc) in enumerate(self.common_formats):
            print(f"   {i + 1}. {desc}")  # desc 格式说明

        while True:
            try:
                choice = input(f"请输入选择（1-{len(self.common_formats)}) : , 或输入'custom'跳过自动推荐：").strip()
                if choice.lower() == 'custom':
                    print("跳过自动推荐")
                    custom_format = input("请输入时间格式(如 %Y-%m-%d): ").strip()
                    # 验证格式
                    try:
                        test_sample = str(samples.iloc[0])
                        self.time_converter.convert_to_datetime(test_sample, time_type='string', format=custom_format)
                        print(f"使用自定义格式：{custom_format}")
                        self.format = custom_format
                        return
                    except:
                        print("格式无效，请重新选择")
                        continue

                else:
                    if choice.isdigit():
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(self.common_formats):
                            selected_format, desc = self.common_formats[choice_idx]
                            print(f"使用格式：{desc}")
                            self.format = selected_format
                            return

                        else:
                            print(f"请输入 1-{len(self.common_formats)} 之间的数字)")
                    else:
                        print("请输入有效的数字")

            except KeyboardInterrupt:
                print("\n用户中断选择")
                return None

    def _smart_format_processing(self, time_series):
        if len(time_series) == 0:
            return
        samples = time_series.dropna()
        if len(samples) == 0:
            print("警告：选择的时间列全是空值")
            return
        print(f"\n正在处理时间列格式format...")
        print(f"样本数量：{len(samples)}")

        # 尝试自动检测格式
        auto_format = self._auto_detect_time_format(time_series)
        if auto_format:
            print(f"自动检测到格式：{auto_format}")
            self.format = auto_format
            return

        # 自动检测format失败，尝试直接pd.to_datetime推断
        print("无法自动检测format格式，尝试直接pd.to_datetime推断...")

        # 测试推断结果
        inferred_result = self.time_converter.convert_to_datetime(samples, time_type='string', format=None)

        success_rate = (1 - inferred_result.isna().mean()) * 100

        if success_rate > 90:
            print(f"推断解析成功率高：{success_rate:.1f}%，使用自动推断")
            self.format = None  # 让pandas自动推断
        else:
            print(f"推断解析成功率低：{success_rate:.1f}%")
            print("样本示例：")
            for i, sample in enumerate(samples.head(5)):
                print(f"  {i + 1}.{sample}")

            # 提供交互式选择
            use_auto = input("是否继续使用to_datetime无format自动解析？(y/n): ").strip().lower()
            if use_auto == 'y':
                self.format = None
                print("使用自动解析")

            else:
                # 调用交互式格式选择
                self.format = self._interactive_select_time_format(time_series)

    def _auto_detect_time_format(self, time_series):
        print("自动解析字符串时间列的format...")
        """自动检测时间格式"""
        if len(time_series) == 0:
            return None
        samples = time_series.dropna().head(5)
        if len(samples) == 0:
            return None

        # 常见时间格式模式
        self.common_formats = [

            # ISO 格式
            ('%Y-%m-%d %H:%M:%S.%f', 'YYYY-MM-DD HH:MM:SS.fff'),
            ('%Y-%m-%d %H:%M:%S', 'YYYY-MM-DD HH:MM:SS'),
            ('%Y-%m-%d %H:%M', 'YYYY-MM-DD HH:MM'),
            ('%Y-%m-%d', 'YYYY-MM-DD'),

            # 斜杠格式
            ('%Y/%m/%d %H:%M:%S', 'YYYY/MM/DD HH:MM:SS'),
            ('%Y/%m/%d %H:%M', 'YYYY/MM/DD HH:MM'),
            ('%Y/%m/%d', 'YYYY/MM/DD'),

            # 美国格式
            ('%m/%d/%Y %H:%M:%S', 'MM/DD/YYYY HH:MM:SS'),
            ('%m/%d/%Y %H:%M', 'MM/DD/YYYY HH:MM'),
            ('%m/%d/%Y', 'MM/DD/YYYY'),

            # 欧洲格式
            ('%d.%m.%Y %H:%M:%S', 'DD.MM.YYYY HH:MM:SS'),
            ('%d.%m.%Y %H:%M', 'DD.MM.YYYY HH:MM'),
            ('%d.%m.%Y', 'DD.MM.YYYY'),
            ('%d/%m/%Y %H:%M:%S', 'DD/MM/YYYY HH:MM:SS'),
            ('%d/%m/%Y %H:%M', 'DD/MM/YYYY HH:MM'),
            ('%d/%m/%Y', 'DD/MM/YYYY'),

            # 中文格式
            ('%Y年%m月%d日 %H时%M分%S秒', 'YYYY年MM月DD日 HH时MM分SS秒'),
            ('%Y年%m月%d日 %H时%M分', 'YYYY年MM月DD日 HH时MM分'),
            ('%Y年%m月%d日', 'YYYY年MM月DD日'),

            # 无分隔符格式
            ('%Y%m%d%H%M%S', 'YYYYMMDDHHMMSS'),
            ('%Y%m%d%H%M', 'YYYYMMDDHHMM'),
            ('%Y%m%d', 'YYYYMMDD'),
        ]

        # 尝试每种格式
        for fmt in self.common_formats:
            format_str = fmt[0]  # 获取实际的格式字符串
            format_desc = fmt[1]  # 获取格式描述（用于调试）
            try:
                test_samples = samples.head(5)
                parsed = self.time_converter.convert_to_datetime(series=test_samples, time_type='string',
                                                                 format=format_str)
                success_rate = parsed.notna().mean()
                print(f"成功率{success_rate:.1%}")
                # 检查是否所有样本都能成功解析且没有NAT

                if not parsed.isna().any():  # 必须保证一个空值都没有，代表‘所有值’均被成功转换
                    print(f"成功匹配格式: {format_desc} ({format_str})")
                    return format_str  # 返回实际的格式字符串

            except:
                continue

        # 如果格式都不匹配，使用dateutil的自动解析如果其能解析，外层的基本pd.to_datetime就一定能解析（更强）
        try:
            from dateutil.parser import parse
            test_sample = str(samples.iloc[0])
            parsed = parse(test_sample)
            print(f"使用 dateutil 可成功解析示例：{test_sample} -> {parsed}")
            return None  # 返回 None 让 pandas 自动推断
        except:
            print(f"使用 dateutil 未成功解析format")
            return None

    def _process_text_time(self, df, col, format):
        # 处理字符串时间 并排好序
        print("转换字符串型时间列...")
        datetime_series = self.time_converter.convert_to_datetime(
            df[col],
            time_type='string',
            format=format
        )

        success_rate = datetime_series.notna().mean()
        if success_rate < 1:
            print(f"警告: 时间列转换成功率 {success_rate:.1%}")

        df[col] = datetime_series
        df = df.sort_values(col, ascending=True)

        print(f"已处理时间字符串列{col}，转成datetime格式")
        return df

    # 提取新的离散特征，后续支持独热编码
    def _new_features_from_timecols(self, df, col):
        print("新增时间特征列...")
        old_cols = df.columns.tolist()

        # 确保列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = self.time_converter.convert_to_datetime(df[col])

        # 气象学中夜间通常指日落到日出的时间段
        hour = df[col].dt.hour
        month = df[col].dt.month

        # 防止空值被转换为false等，只对非空值计算夜间条件
        is_night_series = pd.Series(index=df.index, dtype='object')
        season_type_series = pd.Series(index=df.index, dtype='object')

        # 创建非空掩码
        non_null_mask = df[col].notna()
        if non_null_mask.any():
            hour_non_null = hour[non_null_mask]
            month_non_null = month[non_null_mask]

            # add : is_night 冬季夜间更长，夏季夜间更短
            is_night_basic = (hour_non_null >= 20) | (hour_non_null < 6)

            winter_night = ((month_non_null.isin([12, 1, 2])) & (
                    (hour_non_null >= 19) | (hour_non_null < 7)))  # 冬季（12,1,2月）：晚上7点到早上7点
            summer_night = ((month_non_null.isin([6, 7, 8])) & (
                    (hour_non_null >= 21) | (hour_non_null < 5)))  # 夏季（6,7,8月）：晚上9点到早上5点
            autumn_night = ((month_non_null.isin([9, 10, 11])) & is_night_basic)
            spring_night = ((~month_non_null.isin([12, 1, 2, 6, 7, 8, 9, 10, 11])) & is_night_basic)

            # 将计算结果赋给非空位置(仅判断 is_night 二分的情况，季节已经单独判断）
            night_conditions = winter_night | summer_night | autumn_night | spring_night
            is_night_series[non_null_mask] = night_conditions.astype(bool)

            # add : season 确定季节类型
            winter_mask = month_non_null.isin([12, 1, 2])
            summer_mask = month_non_null.isin([6, 7, 8])
            autumn_mask = month_non_null.isin([9, 10, 11])
            spring_mask = ~(winter_mask | summer_mask | autumn_mask)

            season_type_series[non_null_mask] = np.select(
                [winter_mask, summer_mask, autumn_mask, spring_mask],
                ['winter', 'summer', 'autumn', 'spring'], default='Missing'
            )

            # 空值保持为Nan
            is_night_series[~non_null_mask] = np.nan
            season_type_series[~non_null_mask] = np.nan

        # 转换为分类类型，保持NaN
        df['is_night'] = pd.Categorical(is_night_series)
        df['season'] = pd.Categorical(season_type_series)

        print(f"夜间定义: 冬季(19-7点), 夏季(21-5点), 春秋(20-6点)")
        print(f"df['is_night'].dtype: {df['is_night'].dtype}")
        print(f"df['season_type'].dtype: {df['season'].dtype}")
        print(f"is_night列中空值数量: {df['is_night'].isna().sum()}")

        # add : 时间间隔(数值)
        df['timedelta'] = df[col].diff().dt.total_seconds().fillna(0)

        # add :时间跨度(数值) 对于长期时间序列，可以揭示趋势和周期性（需注意是否为多个独立实体，各自有其起始点）
        df['days_since_start'] = (df[col] - df[col].iloc[0]).dt.days
        df['years_since_start'] = df['days_since_start'] / 365.25

        new_cols = [col for col in df.columns if col not in old_cols]
        print(f"已基于时间列，增加新的特征。包括：{new_cols},共{len(new_cols)}列")
        print(f"目前的数据形状{df.shape}")

        return df

    # 周期编码函数(23点和1点，12月和1月等在循环空间中很接近- season)
    def _cyclic_encoding(self, df, col):
        # 将时刻序列映射为正弦曲线序列)

        # 转换为Unix时间戳秒数,datetime64每个元素都是timestamp实例
        df[col] = (df[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        ts_val = df[col].values

        period_to_seconds = {
            'hour': 60 * 60,
            'day': 24 * 60 * 60,
            'week': 7 * 24 * 60 * 60,
            'month': 30.44 * 24 * 60 * 60,  # 一月多少秒（平均）
            'year': 365.2425 * 24 * 60 * 60
        }

        print("进行周期性编码...")
        df['Day_sin'] = np.sin((ts_val / period_to_seconds['day']) * 2 * np.pi)
        df['Day_cos'] = np.cos((ts_val / period_to_seconds['day']) * 2 * np.pi)
        df['Year_sin'] = np.sin((ts_val / period_to_seconds['year']) * 2 * np.pi)
        df['Year_cos'] = np.cos((ts_val / period_to_seconds['year']) * 2 * np.pi)

        print("新增列'Day_sin'、'Day_cos'、'Day_cos'、'Year_cos'")

        return df


"""处理特殊数据(向量）"""


class ProcessOtherColumns(BaseEstimator, TransformerMixin):
    def __init__(self, dir_cols: Optional[list] = None, var_cols: Optional[list] = None):
        self.dir_cols = dir_cols
        self.var_cols = var_cols

    @validate_input(validate_y=False)
    def fit(self, X, y=None):

        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        if self.dir_cols is None or self.var_cols is None:
            print("未指定待处理vector列，方向 dir_cols 、速度 var_cols 不可为空...")
            return self

        else:
            existing_dir = [col for col in self.dir_cols if col in df.columns]
            missing_dir = [col for col in self.dir_cols if col not in df.columns]
            if missing_dir:
                print(f"警告：指定 dir_cols 列不存在：{missing_dir}")
            if existing_dir:
                self.dir_cols = existing_dir

            existing_var = [col for col in self.var_cols if col in df.columns]
            missing_var = [col for col in self.var_cols if col not in df.columns]

            if missing_var:
                print(f"警告：指定 var_cols 列不存在：{missing_var}")
            if existing_var:
                self.var_cols = existing_var

            return self

    def transform(self, X, y=None):
        print("处理风矢量...")
        """将'风向角度制'和'风速列极坐标'数据转换为风矢量
        dir_cols: 角度值的方向数据，
        var_cols: 极坐标的风速数据"""
        # 处理前:用极坐标（风速m/s）和风向（0-360）来描述风的强度和方向，
        # 处理后:用正交坐标系的两个维度（x轴和y轴）上风的强度，来描述上述'风速'和'风向' ['Wx', 'Wy', 'max Wx', 'max Wy']
        if not self.dir_cols or not self.var_cols:
            print("无'方向'(弧度制)数据 or 无'速度变量'数据需要处理")
            return X
        else:
            X_ = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

            # 处理步骤：
            # a.将风向和风速列数据转换为风矢量，重新存入原数据框中
            # b.2D直方图--通过可视化的方式解释风矢量类型的数据由于原表风速和风向数据的原因

            try:
                # 原表风速和风向数据
                Visualization.plot_hist2d(x=X_[self.dir_cols[0]],  # 'wd'
                                          y=X_[self.var_cols[0]],  # 'wv'
                                          xlabel=f'{self.dir_cols[0]} 风向 [单位：度]',
                                          ylabel=f'{self.var_cols[0]} 风速 [单位：米/秒]')

                # 风矢量类型的数据
                wd_rad = X_.pop(self.dir_cols[0]) * np.pi / 180  # 风向由角度制转换为弧度制

                for i in self.var_cols:
                    value = X_.pop(i)  # 先抓出 再丢了 将df中的wv列保存到wv中，并从原来的df中删除
                    X_[f'Wx_{i}'] = value * np.cos(wd_rad)  # 计算平均风力wv的x和y分量，保存到df的'Wx'列和'Wy'列中
                    X_[f'Wy_{i}'] = value * np.sin(wd_rad)
                    print("新增风矢量数据:")
                    print(X_[[f'Wx_{i}', f'Wy_{i}']].head())

                    # 不需要初始化任何东西，最适合静态方法，然后类名调用
                    Visualization.plot_hist2d(x=X_[f'Wx_{i}'],
                                              y=X_[f'Wy_{i}'],
                                              xlabel='X分量[单位：m/s]',
                                              ylabel='Y分量[单位：m/s]')

                    Visualization.plot_hist2d(x=X_[f'Wx_{i}'],
                                              y=X_[f'Wy_{i}'],
                                              xlabel='X分量[单位：m/s]',
                                              ylabel='Y分量[单位：m/s]')

                if y is not None:
                    return X_, y

            except Exception as e:
                print(f"风矢量处理失败：{e}")
                return X

            return X_


"""处理数值型数据"""  # 可以将时间生成的sin cos 除掉 'timedelta' 'days_since_start' 'year_since_start'


class ProcessNumericColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols: Optional[list] = None,
                 preserve_integer_types: bool = True,
                 exclude_cols: Optional[list] = None):

        self.numeric_columns = cols or []
        self.preserve_integer_types = preserve_integer_types
        self.original_dtypes_ = {}
        self.exclude_cols = exclude_cols or []

        """
        preserve_integer_types:
        object(字符串/混合类型，里面'1', '2', 'abc'] -> [1.0, 2.0, nan]) 默认是float64 -> 改为 [1, 2, nan]
        其他数值型不变 int64，float64
        """

    @validate_input(validate_y=False)
    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        if self.numeric_columns is None:  # None ，空列表[]
            print("未指定待处理数值列，检查原数据的全部数值列...")

            all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            excluded = []
            for col in self.exclude_cols:
                if col in all_numeric_cols:
                    all_numeric_cols.remove(col)
                    excluded.append(col)
            if excluded:
                print(f"已排除时间列：{excluded}")

            self.numeric_columns = all_numeric_cols

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

    @validate_input(validate_y=True)
    def transform(self, X, y=None):
        print("处理数值型数据...")
        if not self.numeric_columns:
            print("无数值列需要处理")
            return X

        X_ = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        for col in self.numeric_columns:
            if col in X_.columns:
                # 保存原类型
                original_dtype = X_[col].dtype

                X_[col] = pd.to_numeric(X_[col], errors='coerce')  # object 不报错，转NaN 默认是float64。

                # 如果标记为保持整数类型、原始是整数类型、转换后没有小数部分，尝试转回整数
                if (self.preserve_integer_types and
                        col in self.original_dtypes_ and
                        np.issubdtype(self.original_dtypes_[col], np.integer)):

                    # 检查是否所有非空值都是整数
                    non_null_values = X_[col].dropna()
                    if len(non_null_values) > 0:
                        # 方法：直接检查小数部分是否为0 .00
                        decimal_parts = non_null_values % 1
                        all_integers = np.all(decimal_parts == 0)  # bool
                        if all_integers:
                            X_[col] = X_[col].astype('Int64')
                print(f"列 {col} 已确认是数值型 (原类型: {original_dtype} -> 现类型: {X_[col].dtype})")
            else:
                print(f"列{col}不在数据中")
                continue

            if y is not None:
                return X_, y

        print("数值型数据处理完成")
        return X_


"""处理分类型/字符串数据"""  # 提前将日期生成的分类列进行astype


class ProcessCategoricalColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols: Optional[list] = None,
                 onehot_threshold: int = 5
                 ):
        self.categorical_columns = cols or []
        self.onehot_threshold = onehot_threshold  # 独热编码的最大类别数阈值
        self.onehot_columns_ = []  # 记录哪些列使用了独热编码

    @validate_input(validate_y=False)
    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        if self.categorical_columns is None:
            print("未指定待处理字符串/分类列，检查原数据的全部字符串/分类列...")
            self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if not self.categorical_columns:
                print("数据无分类型/字符串型列需处理")
            else:
                print(f"自动识别分类型/字符串型列:{self.categorical_columns}")

        else:
            existing_cols = [col for col in self.categorical_columns if col in df.columns]
            missing_cols = [col for col in self.categorical_columns if col not in df.columns]
            if missing_cols:
                print(f"警告: 以下指定列不存在:{missing_cols}")
            self.categorical_columns = existing_cols
            print(f"使用指定列: {self.categorical_columns}")

        # 确定哪些列使用独热编码
        for col in self.categorical_columns:
            unique_count = df[col].nunique()  # Excludes NA values by default.
            if unique_count <= self.onehot_threshold:
                self.onehot_columns_.append(col)
                print(f"列 '{col}' 将使用独热编码 (唯一值数量: {unique_count})")

        return self

    @validate_input(validate_y=True)
    def transform(self, X, y=None):
        print("处理分类型/字符串数据...")

        if self.categorical_columns is None:
            print("无分类型/字符串列需处理")
            return X

        X_ = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        for col in self.categorical_columns:
            # 处理字符串类型的时间分列
            if col == 'Date Time':
                # object 转 datetime
                X_ = self._strptime(X_, col)
            if col in self.onehot_columns_:
                X_ = self._onehot(X_, col)

        if y is not None:
            return X_, y
        return X_

    def _strptime(self, df, col):
        # 处理字符串时间 并排好序
        datetime = pd.to_datetime(df.pop(col), format='%d.%m.%Y %H:%M:%S')
        df[col] = datetime
        df = df.sort_values(col, ascending=True)
        print(f"已处理时间字符串列{col}，转成datetime格式")
        return df

    def _onehot(self, df, col):
        # 处理分类
        # 1.分类数量少，四季(独热编码)
        # 2.分类数量多，电商的产品ID、店铺ID，模型内嵌入层 (Embedding Layer)，将高基数分类特征转换为密集向量表示
        # 即使输入已经处理，如果是预测分类变量，也要处理输出层激活函数以及损失函数。而且layers也是需要分开卷积再合并！

        try:
            encoded_df = pd.get_dummies(df[col], prefix=col)
            # 删除原始列
            df = df.drop(col, axis=1)

            # 合并编码后的列
            df = pd.concat([df, encoded_df], axis=1)
            print(f"列 '{col}' 独热编码完成，新增 {len(encoded_df.columns)} 个特征")

        except Exception as e:
            print(f"列 '{col}' 独热编码失败: {str(e)}")

        return df



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



"""处理缺失值"""


class HandleMissingValue(BaseEstimator, TransformerMixin):
    def __init__(self, cat_strategy: str = 'custom',  # 支持众数填充/自定义Missing填充
                 num_strategy: str = 'mean',
                 num_fill_value=None):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


def handle_missing_values(self,
                          cat_strategy: str = 'custom',
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
