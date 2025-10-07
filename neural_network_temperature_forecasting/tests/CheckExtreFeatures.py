import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from functools import wraps
from sklearn.utils import check_array


# 复制您的装饰器代码
def validate_input(validate_y=True, allow_empty=False, **param_checks):
    """输入验证装饰器"""

    def decorator(method):
        @wraps(method)
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


# 复制您的 CheckExtreFeatures 类代码
class CheckExtreFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, method_config=None):
        if method_config is None:
            self.method_config = {'method': 'iqr', 'threshold': 1.5}
        else:
            self.method_config = dict(method_config)

        self.stats_ = {}
        self.numeric_columns = []
        self.outliers_details = pd.DataFrame()
        self.outliers_info = []

    @validate_input(validate_y=False)
    def fit(self, X, y=None):
        self.method = self.method_config.get('method', 'iqr').lower()

        if self.method not in ['iqr', 'zscore']:
            print(f"不支持的异常值查看方法：{self.method}。使用默认iqr(阈值1.5)的配置查看。")
            self.method = 'iqr'
            self.threshold = 1.5
        else:
            threshold = self.method_config.get('threshold')
            if threshold is None:
                threshold_map = {'zscore': 3, 'iqr': 1.5}
                self.threshold = threshold_map.get(self.method)
            else:
                self.threshold = threshold
            print(f"将使用{self.method}方法标记每列异常值，阈值{self.threshold}")

        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in self.numeric_columns:
            clean_series = df[col].dropna()
            if len(clean_series) < 2:
                print(f"列{col}数据不足，跳过所有方法统计")
                continue

            if self.method == 'zscore':
                self.stats_[col] = {
                    'mean': clean_series.mean(),
                    'std': clean_series.std(),
                    'method': 'zscore'
                }
            elif self.method == 'iqr':
                if len(clean_series) < 4:
                    print(f"列{col}数据不足，跳过iqr方法统计")
                    continue

                q1, q3 = clean_series.quantile([0.25, 0.75])
                iqr = q3 - q1
                self.stats_[col] = {
                    'q1': q1,
                    'q3': q3,
                    'iqr': iqr,
                    'method': 'iqr'
                }
                if iqr == 0:
                    print(f"列'{col}': iqr为0，数据高度集中，大部分值相同，只有少数离群值")
        return self

    @validate_input(validate_y=True)
    def transform(self, X, y=None):
        if not hasattr(self, 'stats_'):
            raise ValueError("请先调用fit方法进行配置")

        print("标记或替换NaN数值列的异常值...")
        X_ = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        all_outliers_list = []
        outliers_info = []

        for col, stats in self.stats_.items():
            if col not in X_.columns:
                continue

            clean_series = X_[col].dropna()
            if len(clean_series) < 2:
                continue

            if stats['method'] == 'zscore':
                z_scores = np.abs((clean_series - stats['mean']) / stats['std'])
                outlier_mask = (z_scores >= self.threshold)

                self._collect_extre_features(
                    df=X_, col=col, series=clean_series, mask=outlier_mask,
                    scores=z_scores, outliers_list=all_outliers_list,
                    info_list=outliers_info
                )

            elif stats['method'] == 'iqr':
                iqr = stats['iqr']
                lower_bound = stats['q1'] - self.threshold * iqr
                upper_bound = stats['q3'] + self.threshold * iqr
                outlier_mask = (clean_series < lower_bound) | (clean_series > upper_bound)

                self._collect_extre_features(
                    df=X_, col=col, series=clean_series, mask=outlier_mask,
                    scores=clean_series, outliers_list=all_outliers_list,
                    info_list=outliers_info
                )

        self.outliers_info = outliers_info
        self.outliers_details = self._format_results(outliers_list=all_outliers_list)

        outliers_mask = np.zeros(X_.shape[0], dtype=bool)
        if not self.outliers_details.empty:
            outliers_mask[self.outliers_details.index] = True

        X_['is_outliers'] = outliers_mask

        if y is not None:
            return X_, y
        return X_

    def _collect_extre_features(self, df, col, series, mask, scores, outliers_list, info_list):
        outlier_indices = series.index[mask]

        if mask.any():
            outlier_df = df.loc[outlier_indices].copy()
            outlier_df['outlier_source'] = col
            outlier_df['score'] = scores[mask]
            outlier_df['original_index'] = outlier_indices

            outliers_list.append(outlier_df)
            info_list.append({
                'column': col,
                'method': self.method,
                'outlier_count': len(outlier_df),
                'threshold': self.threshold
            })

    def _format_results(self, outliers_list):
        if outliers_list:
            df = pd.concat(outliers_list, ignore_index=False)
            grouped = df.groupby('original_index').agg({
                'outlier_source': lambda x: list(x),
                'score': lambda x: list(x)
            }).rename(columns={
                'outlier_source': 'extreme_tag',
                'score': 'extreme_scores'
            })
            return grouped
        else:
            return pd.DataFrame()


# 修复后的测试函数
def test_check_extre_features():
    print("=" * 60)
    print("开始测试 CheckExtreFeatures 类")
    print("=" * 60)

    # 创建测试数据 - 修复长度不一致问题
    np.random.seed(42)
    n_samples = 100

    # 所有数组长度必须相同
    data = {
        'normal_col': np.random.normal(0, 1, n_samples),
        'skewed_col': np.concatenate([np.random.exponential(2, n_samples - 5), [50, 60, 70, 80, 90]]),  # 添加异常值
        'constant_col': np.concatenate([np.ones(n_samples - 3), [100, 200, 300]]),  # 常数列+异常值
        'short_col': np.concatenate([np.random.normal(0, 1, 3), [100], np.full(n_samples - 4, np.nan)])
        # 修复：使用NaN填充到相同长度
    }

    df = pd.DataFrame(data)
    y = np.random.randint(0, 2, n_samples)  # 虚拟目标变量

    print("测试数据:")
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print("\n数据统计:")
    print(df.describe())
    print("\n每列非空值数量:")
    print(df.count())
    print("\n" + "=" * 40)

    # 测试 1: 默认配置 (IQR)
    print("\n测试 1: 默认 IQR 方法")
    detector_iqr = CheckExtreFeatures()
    try:
        X_transformed_iqr = detector_iqr.fit_transform(df)
        print("✓ IQR 检测器成功运行")
        print(f"检测到异常值数量: {X_transformed_iqr['is_outliers'].sum()}")
        print(f"异常值详情形状: {detector_iqr.outliers_details.shape}")
        if not detector_iqr.outliers_details.empty:
            print("异常值详情 (前5行):")
            print(detector_iqr.outliers_details.head())
        print(f"异常信息: {detector_iqr.outliers_info}")
    except Exception as e:
        print(f"✗ IQR 检测器失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 40)

    # 测试 2: Z-score 方法
    print("\n测试 2: Z-score 方法")
    try:
        detector_zscore = CheckExtreFeatures({'method': 'zscore', 'threshold': 3})
        X_transformed_zscore = detector_zscore.fit_transform(df)
        print("✓ Z-score 检测器成功运行")
        print(f"检测到异常值数量: {X_transformed_zscore['is_outliers'].sum()}")
        print(f"异常值详情形状: {detector_zscore.outliers_details.shape}")
        if not detector_zscore.outliers_details.empty:
            print("异常值详情 (前5行):")
            print(detector_zscore.outliers_details.head())
    except Exception as e:
        print(f"✗ Z-score 检测器失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 40)

    # 测试 3: 带 y 参数
    print("\n测试 3: 带目标变量 y")
    try:
        detector_with_y = CheckExtreFeatures()
        X_transformed, y_returned = detector_with_y.fit_transform(df, y)
        print("✓ 带 y 参数检测器成功运行")
        print(f"返回的 X 形状: {X_transformed.shape}")
        print(f"返回的 y 形状: {y_returned.shape}")
        print(f"检测到异常值数量: {X_transformed['is_outliers'].sum()}")
    except Exception as e:
        print(f"✗ 带 y 参数检测器失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 40)

    # 测试 4: 装饰器验证功能
    print("\n测试 4: 装饰器输入验证")

    # 测试空数据
    print("测试空数据验证:")
    try:
        empty_detector = CheckExtreFeatures()
        empty_detector.fit(np.array([]))
        print("✗ 空数据验证失败 - 应该抛出异常")
    except ValueError as e:
        print(f"✓ 空数据验证成功: {e}")

    # 测试 None 数据
    print("\n测试 None 数据验证:")
    try:
        none_detector = CheckExtreFeatures()
        none_detector.fit(None)
        print("✗ None 数据验证失败 - 应该抛出异常")
    except ValueError as e:
        print(f"✓ None 数据验证成功: {e}")

    # 测试类型错误
    print("\n测试类型错误验证:")
    try:
        type_detector = CheckExtreFeatures()
        type_detector.fit("invalid_type")
        print("✗ 类型验证失败 - 应该抛出异常")
    except TypeError as e:
        print(f"✓ 类型验证成功: {e}")

    # 测试 X y 长度不一致
    print("\n测试 X y 长度不一致:")
    try:
        length_detector = CheckExtreFeatures()
        wrong_y = np.array([1, 2, 3])  # 长度与 X 不同
        length_detector.fit_transform(df, wrong_y)
        print("✗ 长度验证失败 - 应该抛出异常")
    except ValueError as e:
        print(f"✓ 长度验证成功: {e}")

    print("\n" + "=" * 40)

    # 测试 5: 分别调用 fit 和 transform
    print("\n测试 5: 分别调用 fit 和 transform")
    try:
        separate_detector = CheckExtreFeatures()
        separate_detector.fit(df)
        print("✓ fit 方法成功")

        X_separate = separate_detector.transform(df)
        print("✓ transform 方法成功")
        print(f"检测到异常值数量: {X_separate['is_outliers'].sum()}")

        # 测试在没有 fit 的情况下调用 transform
        print("\n测试未 fit 直接 transform:")
        unfitted_detector = CheckExtreFeatures()
        try:
            unfitted_detector.transform(df)
            print("✗ 未 fit 验证失败 - 应该抛出异常")
        except ValueError as e:
            print(f"✓ 未 fit 验证成功: {e}")

    except Exception as e:
        print(f"✗ 分别调用失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


# 运行测试
if __name__ == "__main__":
    test_check_extre_features()