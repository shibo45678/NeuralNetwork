from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin

class SmartOutlierProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config_name='default'):
        self.config_name = config_name
        self.detection_functions = self._setup_detection_functions() # 初始化即可完成所有配置

    def _setup_detection_functions(self):
        """基于配置名设置不同的检测函数"""
        if self.config_name == 'strict':
            return self._create_strict_handlers()
        else:
            return self._create_default_handlers()

    def _create_strict_handlers(self):
        """创建严格的异常值检测器"""

        def zscore_detector(series):
            z_scores = np.abs((series - series.mean()) / series.std())
            return z_scores > 2  # 判断条件

        def iqr_detector(series):
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            return (series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))

        return {
            'zscore': zscore_detector, # 返回闭包函数
            'iqr': iqr_detector
        }

    def _create_default_handlers(self):
        """创建默认的异常值检测器"""
        # 类似的实现...
        pass

    def fit(self, X, y=None):
        # 可以基于训练数据调整参数
        print(f"使用 {self.config_name} 配置进行拟合")
        return self

    def transform(self, X):
        X_processed = X.copy()
        outlier_masks = {}

        for col in X.columns:
            for method_name, detector in self.detection_functions.items():
                mask = detector(X[col])
                outlier_masks[f"{col}_{method_name}"] = mask
                # 在实际应用中，这里会有处理逻辑

        print(f"处理了 {len(outlier_masks)} 种异常值检测")
        return X_processed


# 创建完整的 Pipeline
pipeline = Pipeline([
    ('outlier_detection', SmartOutlierProcessor(config_name='strict')),
])

# 使用 Pipeline
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=4, random_state=42)

# 拟合和转换 - 所有参数都在类内部处理！
X = pd.DataFrame(X)
pipeline.fit(X)
result = pipeline.transform(X)

from sklearn.base import BaseEstimator, TransformerMixin
from data.data_preparation.check_extre_numeric_features import CheckExtreFeatures
from data.decorator import validate_input, validate_output
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Union, Any
from sklearn.ensemble import IsolationForest
import logging

logger = logging.getLogger(__name__)
"""处理异常值"""

"""____________________________________________________________________________"""
class NumericOutlierProcessor:
    """异常值处理器"""

    def __init__(self,
                 method_config: Union[Dict[str, Any], List[Tuple[str, Dict]]] = None,
                 handle_strategy: str = 'remove',
                 pass_through=False):
        """
        Args:
            method_config: 可以是字典(全局配置)或列表(列级配置,不同列不同的method)。
                           其中列级配置包含自定义函数custom_handlers: Optional[Dict[str, Callable]] = None;
                           仅支持二选一；
            handle_strategy:异常值处理策略 mark/remove/impute插值/设为nan后续填空值/constant/custom.
                            **remove不能进pipeline，因为相当于remove_rows，其他可以选择子类实现fit_transform

            1. 全局配置
               {'method': 'isolationforest','threshold':0.025}
               隔离森林，联合检测，在check_extre_features.py设置

            2. 列表(列级配置)
               [('zscore' ,       {'threshold':3,             'columns':[]}),
                ('iqr'    ,       {'threshold':1.5,           'columns':[]}),
                ('robust' ,        {'quantile_range':(25, 75), 'columns':[]), # 分位数
                ('isolationforest',{'contamination':0.025,    'columns':[]})
                ('minmax' ,        {'feature_range':[(0,1),(0,100)...],'columns':[column1,column2...]} )  # 数值范围检测：明确业务规则或物理约束}
                ('custom',         {'functions':[func1,func2...],        columns':[]}) # 传递函数
                ] 不同列选择不一样的异常值查找方式;每列只能有一种列级配置;

            3. 填充方式
               全局共用一种检测的无法用插值填充no impute；
        """

        self.method_config = method_config
        self.pass_through = pass_through
        self.handle_strategy = handle_strategy
        self.stats_ = {}
        self.detector_ = None
        self.outlier_mask_ = None  # 存储所有列异常的并集
        self.column_detectors_ = {}  # 有存储逐列异常判断的结果
        self.outlier_flag_columns_ = []  # 主要针对标记mark 异常值处理器的列

    @validate_input(validate_y=True)
    def learn(self, X, y=None):
        if self.pass_through:
            return self
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        # 初始化异常值掩码
        self.outlier_mask_ = pd.Series(False, index=df.index)

        # 1 处理全局配置
        if isinstance(self.method_config, dict):
            self._fit_global_same_detection(df, y)

        # 2 处理列级配置
        elif isinstance(self.method_config, list):
            self._fit_column_diff_detection(df, y)

        return self

    @validate_input(validate_y=True)
    @validate_output(validate_y=True)
    def process(self, X, y=None):
        if self.pass_through:
            return X, y
        if self.outlier_mask_ is None:
            raise ValueError("请先调用fit方法")

        X_ = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        y_ = pd.Series(y) if y is not None else None

        logger.debug(f"应用异常值处理策略: {self.handle_strategy}")

        if self.handle_strategy == 'remove':  # 不可进pipeline
            return self._apply_remove_strategy(X_, y_)
        elif self.handle_strategy == 'mark':
            return self._apply_mark_strategy(X_)
        elif self.handle_strategy == 'nan':
            return self._apply_nan_strategy(X_)
        elif self.handle_strategy == 'custom':
            return self._apply_custom_strategy(X_)
        elif self.handle_strategy == 'impute':
            if isinstance(self.method_config, list):
                return self._apply_impute_strategy(X_)
            else:
                raise ValueError("只有逐列检测才能进行中位数插值，修改 method_config")
        else:
            raise ValueError(f"不支持的策略: {self.handle_strategy}")

    def _fit_global_same_detection(self, X: pd.DataFrame, y=None):
        """使用全局检测器"""
        logger.debug("使用全局异常值检测器...")

        self.detector_ = CheckExtreFeatures(method_config=self.method_config)
        self.detector_.fit(X, y)
        _ = self.detector_.transform(X, y)  # 仅需要 outliers_details

        if hasattr(self.detector_, 'outliers_details') and not self.detector_.outliers_details.empty:
            global_outlier_mask = pd.Series(False, index=X.index)
            common_indices = self.detector_.outliers_details.index.intersection(X.index)

            global_outlier_mask.loc[common_indices] = True
            # 转换为numpy数组并合并到整体掩码
            self.outlier_mask_ = global_outlier_mask

    def _fit_column_diff_detection(self, X: pd.DataFrame, y: pd.Series = None):
        """为不同列配置不同的检测器"""
        logger.debug("使用列级异常值检测器...")
        for config in self.method_config:
            method_name, params = config
            columns = params.get('columns', [])

            if not columns:  # 如果没有指定列，就不处理
                continue

            for idx, col in enumerate(columns):
                if col not in X.columns:
                    continue

                series_clean = X[col].dropna()

                if method_name == 'zscore':
                    mask = self._zscore_detector(series_clean, params)
                elif method_name == 'iqr':
                    mask = self._iqr_detector(series_clean, params)
                elif method_name == 'robust':
                    mask = self._robust_detector(series_clean, params)
                elif method_name == 'isolationforest':
                    mask = self._isolation_detector(series_clean, params)
                elif method_name == 'minmax':  # 这里并不是正常归一，不可复用至标准化
                    mask = self._minmax_detector(series_clean, params, idx)
                elif method_name == 'custom':
                    mask = self._custom_detector(series_clean, params, idx)

                else:
                    logger.debug(f"暂不支持的方法{method_name}")
                    continue

                if mask is None:  # 特别处理返回None的情况
                    logger.debug(f"列{col}的{method_name}检测器返回的None，未检测到异常值，跳过该列")
                    continue

                # 1.确保 mask是pandas Series
                if isinstance(mask, np.ndarray):
                    mask = pd.Series(mask, index=series_clean.index)  # 也是dropna的

                # 2.还原，确保长度是原列长度 非dropna后的
                full_mask = pd.Series(False, index=X.index)
                full_mask.loc[series_clean.index[mask]] = True

                self.outlier_mask_ = self.outlier_mask_ | full_mask

                if col not in self.column_detectors_:
                    self.column_detectors_[col] = []  # 防止1列多种配置，后续默认选择第一个出现的方式

                # 规则：逐列操作一个列只允许一种异常值操作
                self.column_detectors_[col].append({
                    'method': method_name,
                    'outlier_count': mask.sum(),
                    'single_mask': full_mask
                })

    def _zscore_detector(self, series: pd.Series, params: Dict):
        """Z-score检测器"""
        threshold = params.get('threshold', 3)

        if len(series) >= 2:  # 不满足的也会返回值 None
            z_score = np.abs(stats.zscore(series))
            return z_score > threshold
        else:
            logger.debug(f"数据不足，跳过Z-score检测")
            return pd.Series(False, index=series.index)

    def _iqr_detector(self, series: pd.Series, params: Dict):
        threshold = params.get('threshold', 1.5)

        if len(series) >= 4:
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            return (series < lower_bound) | (series > upper_bound)
        else:
            logger.debug(f"数据不足，跳过iqr检测")
            return pd.Series(False, index=series.index)

    def _robust_detector(self, series: pd.Series, params: Dict):
        """Robust检测器（基于分位数）"""
        quantile_range = params.get('quantile_range', (10, 90))
        lower_q, upper_q = quantile_range

        if len(series) >= 2:
            lower_bound = np.percentile(series, lower_q)
            upper_bound = np.percentile(series, upper_q)
            return (series > upper_bound) | (series < lower_bound)

        else:
            logger.debug(f"数据不足，跳过robust检测")
            return pd.Series(False, index=series.index)

    def _isolation_detector(self, series: pd.Series, params: Dict):
        """隔离森林检测器:联合检测"""
        contamination = params.get('contamination', 0.025)  # 预期异常比例
        model = IsolationForest(contamination=contamination, random_state=42)
        predictions = model.fit_predict(series.values.reshape(-1, 1))  # -1表示异常

        return pd.Series(predictions == -1, index=series.index)

    def _minmax_detector(self, series: pd.Series, params: Dict, idx: int):
        """min-max范围检测器"""

        feature_range = params.get('feature_range', [(0, 1)])  # 根据业务数值
        min_val, max_val = feature_range[
            idx]  # 格式与其他稍有不同 params {'feature_range':[(0,1),(0,100)...],'columns':[column1,column2...]}

        return (series < min_val) | (series > max_val)

    def _custom_detector(self, series: pd.Series, params: Dict, idx: int):
        """执行自定义异常值检测(物理意义)"""
        logger.debug("执行自定义异常值检测...")  # 格式与其他稍有不同 params {'functions':[func1,func2...],        columns':[]}

        func_list = params.get('functions', [])
        func = func_list[idx]

        try:
            # 调用自定义处理函数
            return func(series)
        except Exception as e:
            logger.debug(f"自定义检测列失败：{e}")

    def _apply_remove_strategy(self, X_: pd.DataFrame, y_: pd.Series) -> Union[
        Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
        # 全局是outlier_mask_,逐列也是outlier_mask_，都是整体的并集； 变动了数据集，不可进pipeline；

        normal_mask = ~self.outlier_mask_
        X_cleaned = X_.loc[normal_mask.values]

        if y_ is not None:
            y_cleaned = y_.loc[normal_mask.values]
            result = (X_cleaned, y_cleaned)
        else:
            result = X_cleaned

        logger.debug("应用移除处理策略")
        return result

    def _apply_mark_strategy(self, X_: pd.DataFrame) -> pd.DataFrame:
        # 未变动数据集 可连接pipeline
        if isinstance(self.method_config, dict):  # 全局只添加一列
            X_['is_entire_outlier'] = self.outlier_mask_
            # 转换为分类变量 ,对于树模型可以直接使用，对于线性模型建议独热编码
            X_['is_entire_outlier'] = X_['is_entire_outlier'].astype('int')

        elif isinstance(self.method_config, list):
            # 逐列添加各列的异常值标记
            for col, info in self.column_detectors_.items():
                if col in X_.columns:
                    if len(info) > 1:
                        logger.debug(f"列{col}存在多种异常值检测方式，自动选择第一个方式{info[0]['method']}")

                    method = info[0]['method']
                    mask = info[0]['single_mask']
                    if hasattr(mask, 'values'):
                        mask = mask.values
                        if method in ['custom', 'isolationforest', 'robust', 'iqr', 'minmax', 'zscore']:
                            X_[f'is_outlier_{col}'] = False
                            X_.loc[mask, f'is_outlier_{col}'] = True
                            X_[f'is_outlier_{col}'] = X_[f'is_outlier_{col}'].astype('category')

        self.outlier_flag_columns_ = [col for col in X_.columns if
                                      col.startswith('is_outlier_') or col.startswith('is_entire_outlier')]
        logger.debug(f"应用标记处理策略，添加了异常值列，并转为分类列")
        return X_

    def _apply_impute_strategy(self, X_: pd.DataFrame) -> pd.DataFrame:
        # 逐列操作；对异常值进行中位数插值（全局不可）
        if isinstance(self.method_config, list):
            X_imputed = X_.copy()

            # 处理列级检测器id x异常值
            for col, info in self.column_detectors_.items():
                if col in X_imputed.columns:
                    if len(info) >= 1:
                        logger.debug(f"列{col}存在多种异常值检测方式，自动选择第一个方式{info[0]['method']}")

                    method = info[0]['method']
                    mask = info[0]['single_mask']
                    if hasattr(mask, 'values'):
                        mask = mask.values
                        if method in ['custom', 'isolationforest', 'robust', 'iqr', 'minmax', 'zscore']:
                            if mask.sum() > 0:
                                median_val = X_imputed[col].median()
                                X_imputed.loc[mask, col] = median_val

            logger.debug("应用插值处理策略")
            return X_imputed

        else:
            logger.debug("全局异常值不支持逐列插值")
            return X_

    def _apply_nan_strategy(self, X_: pd.DataFrame) -> pd.DataFrame:  # 逐列操作

        X_nan = X_.copy()
        if isinstance(self.method_config, list):
            for col, info in self.column_detectors_.items():
                if col in X_nan.columns:

                    if len(info) >= 1:
                        logger.debug(f"列{col}存在多种异常值检测方式，自动选择第一个方式{info[0]['method']}")

                    method = info[0]['method']
                    mask = info[0]['single_mask']
                    if hasattr(mask, 'values'):
                        mask = mask.values
                        if method in ['custom', 'isolationforest', 'robust', 'iqr', 'minmax', 'zscore']:

                            if mask.sum() > 0:
                                X_nan.loc[mask, col] = np.nan

        else:  # 空method_config + dict 的情况
            if hasattr(self.detector_, 'outlier_mask') and self.detector_.outlier_mask is not None:
                X_nan.loc[self.detector_.outlier_mask] = np.nan  # 整行变成nan

        logger.debug("应用nan处理策略")
        return X_nan

    def _apply_custom_strategy(self, X_: pd.DataFrame) -> pd.DataFrame:
        """
        应用自定义异常值处理策略
        将自定义检测到的异常值设为0，并更新相关状态
        """
        X_custom = X_.copy()
        logger.debug("执行自定义异常值处理...")

        for col, info in self.column_detectors_.items():
            if col == 'wv':

                if len(info) >= 1:
                    logger.debug(f"列{col}存在多种异常值检测方式，自动选择第一个方式{info[0]['method']}")

                method = info[0]['method']
                mask = info[0]['single_mask']
                if hasattr(mask, 'values'):
                    mask = mask.values
                    if method in ['custom', 'isolationforest', 'robust', 'iqr', 'minmax', 'zscore']:
                        if mask.sum() > 0:
                            X_custom.loc[mask, col] = 0

        logger.debug("应用自定义异常值处理策略")
        return X_custom

    def get_detection_report(self) -> Dict:
        if self.outlier_mask_ is None:
            return {"error": "请先调用fit方法"}

        report = {
            'total_outliers': self.outlier_mask_.sum(),
            'handle_strategy': self.handle_strategy,
            'global_detector': None,
            'column_detectors': self.column_detectors_,
        }
        if self.detector_:  # 统一检测器结果报告
            report['global_detector'] = {
                'method': getattr(self.detector_, 'method', 'unknown'),
                'threshold': getattr(self.detector_, 'threshold', 'unknown'),
                'affected_features': len(getattr(self.detector_, 'outlier_info', {}))
            }
        return report


def handler(series, threshold=0):
    return series < threshold


class CustomNumericOutlier(NumericOutlierProcessor, BaseEstimator, TransformerMixin):
    def __init__(self, method_config, pass_through=False, handle_config: dict = None):
        NumericOutlierProcessor.__init__(self, method_config, handle_strategy='custom', pass_through=pass_through)
        self.handle_config = handle_config

    # @validate_input(validate_y=False)
    def fit(self, X, y=None):
        # logger.debug(f"CustomNumericOutlier.fit: 数据形状 {X.shape}")
        self.learn(X, y)
        return self

    # @validate_input(validate_y=False)
    def transform(self, X):
        return self.process(X)

    def _apply_custom_strategy(self, data: pd.DataFrame):
        """
        仅将异常值里面的wv,max. wv 判断为异常值的全部替换成0。其他列异常值没有配置，全部按照mark处理
        """
        handle_dict = self._parse_config()

        for col, info in self.column_detectors_.items():
            if col in data.columns:

                if len(info) >= 1:
                    logger.debug(f"列{col}存在多种异常值检测方式，自动选择第一个方式{info[0]['method']}")

                mask = info[0]['single_mask']
                handle_method = handle_dict.get(col, None)
                if mask.sum() > 0:
                    if hasattr(mask, 'values'):
                        mask = mask.values  # 转成 numpy array
                        if handle_method == 'custom':
                            data.loc[mask, col] = 0
                        elif handle_method == 'nan':
                            data.loc[mask, col] = np.nan
                        else:
                            data[f'is_outlier_{col}'] = 0
                            data.loc[mask, f'is_outlier_{col}'] = 1
                            data[f'is_outlier_{col}'] = data[f'is_outlier_{col}'].astype('int')

        self.outlier_flag_columns_ = [col for col in data.columns if
                                      col.startswith('is_outlier_') or col.startswith('is_entire_outlier')]
        return data

    def _parse_config(self):
        dict = {}
        for method, columns in self.handle_config.items():
            for col in columns:
                dict[col] = method
        return dict

# 放进learn_process里面 每个数据集单独处理