from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin


class SmartOutlierDetector(BaseEstimator, TransformerMixin):
    def __init__(self, auto_select=True, default_method='iqr',
                 skew_threshold=1.0, kurtosis_threshold=3.0):
        """
        参数:
        auto_select: 是否自动选择最佳方法
        default_method: 自动选择失败时的默认方法 ('zscore', 'iqr', 'robust')
        skew_threshold: 偏度阈值，超过则认为数据偏态
        kurtosis_threshold: 峰度阈值，超过则认为数据重尾
        """
        self.auto_select = auto_select
        self.default_method = default_method
        self.skew_threshold = skew_threshold
        self.kurtosis_threshold = kurtosis_threshold
        self.selected_methods_ = {}
        self.detection_stats_ = {}

    def _analyze_distribution(self, data):
        """分析数据分布特征"""
        if len(data) < 10:  # 数据太少，无法可靠分析
            return {'method': 'iqr', 'reason': '数据量太少'}

        # 计算统计量
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        normality_p = stats.normaltest(data).pvalue if len(data) > 20 else 0

        stats_dict = {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'normality_p': normality_p,
            'is_normal': normality_p > 0.05,
            'is_skewed': abs(skewness) > self.skew_threshold,
            'is_heavy_tailed': kurtosis > self.kurtosis_threshold,
            'n_samples': len(data)
        }

        return stats_dict

    def _select_best_method(self, data, column_name):
        """为单个列选择最佳异常值检测方法"""
        stats_info = self._analyze_distribution(data)

        # 决策逻辑
        if stats_info['n_samples'] < 10:
            method = 'iqr'
            reason = '数据量少，使用稳健的IQR方法'

        elif stats_info['is_normal'] and not stats_info['is_skewed']:
            method = 'zscore'
            reason = '数据接近正态分布，使用Z-score方法'

        elif stats_info['is_skewed']:
            method = 'iqr'
            reason = f"数据偏态(偏度={stats_info['skewness']:.2f})，使用IQR方法"

        elif stats_info['is_heavy_tailed']:
            method = 'robust'
            reason = f"数据重尾(峰度={stats_info['kurtosis']:.2f})，使用稳健Z-score方法"

        else:
            method = self.default_method
            reason = f'使用默认方法: {self.default_method}'

        return {
            'method': method,
            'reason': reason,
            'stats': stats_info
        }

    def fit(self, X, y=None):
        """拟合数据，为每个列选择最佳方法"""
        self.n_features_in_ = X.shape[1] if hasattr(X, 'shape') else len(X.columns)

        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()

        self.selected_methods_ = {}
        self.detection_stats_ = {}

        for col in X.columns:
            data = X[col].dropna()
            if len(data) == 0:
                continue

            if self.auto_select:
                # 自动选择方法
                result = self._select_best_method(data, col)
                self.selected_methods_[col] = result['method']
                self.detection_stats_[col] = result
            else:
                # 使用默认方法
                self.selected_methods_[col] = self.default_method
                self.detection_stats_[col] = {
                    'method': self.default_method,
                    'reason': '使用用户指定的默认方法',
                    'stats': self._analyze_distribution(data)
                }

        return self

    def _detect_outliers_zscore(self, data, threshold=3):
        """Z-score 方法检测异常值"""
        z_scores = np.abs(stats.zscore(data))
        return z_scores > threshold

    def _detect_outliers_iqr(self, data, threshold=1.5):
        """IQR 方法检测异常值"""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (data < lower_bound) | (data > upper_bound)

    def _detect_outliers_robust(self, data, threshold=3):
        """稳健Z-score方法，使用中位数和MAD"""
        median = np.median(data)
        mad = stats.median_abs_deviation(data)
        if mad == 0:  # 避免除零
            mad = np.median(np.abs(data - median)) * 1.4826
        robust_z = np.abs(0.6745 * (data - median) / mad)
        return robust_z > threshold

    def transform(self, X):
        """检测异常值"""
        check_is_fitted(self, 'n_features_in_')

        outlier_results = {}

        for col in X.columns:
            if col not in self.selected_methods_:
                continue

            data = X[col].dropna()
            if len(data) == 0:
                outlier_results[col] = np.full(len(X), False)
                continue

            method = self.selected_methods_[col]

            if method == 'zscore':
                is_outlier = self._detect_outliers_zscore(data)
            elif method == 'iqr':
                is_outlier = self._detect_outliers_iqr(data)
            elif method == 'robust':
                is_outlier = self._detect_outliers_robust(data)
            else:
                # 回退到IQR
                is_outlier = self._detect_outliers_iqr(data)

            # 创建与原始数据相同长度的结果
            full_is_outlier = np.full(len(X), False)
            full_is_outlier[data.index] = is_outlier
            outlier_results[col] = full_is_outlier

        return outlier_results

    def get_detection_report(self):
        """获取检测方法选择报告"""
        if not hasattr(self, 'selected_methods_'):
            return "模型尚未拟合"

        report = "异常值检测方法选择报告:\n"
        report += "=" * 50 + "\n"

        for col, method in self.selected_methods_.items():
            stats_info = self.detection_stats_[col]
            report += f"列: {col}\n"
            report += f"  选择方法: {method}\n"
            report += f"  原因: {stats_info['reason']}\n"
            report += f"  统计量 - 偏度: {stats_info['stats']['skewness']:.2f}, "
            report += f"峰度: {stats_info['stats']['kurtosis']:.2f}, "
            report += f"正态性p值: {stats_info['stats']['normality_p']:.3f}\n\n"

        return report



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