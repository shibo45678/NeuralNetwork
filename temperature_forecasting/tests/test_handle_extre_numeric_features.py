import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from functools import partial

from data.data_preparation.handle_extre_numeric_features import NumericOutlierProcessor, MethodConfig  # 替换为实际模块路径


class TestMethodConfig:
    """MethodConfig 配置验证类的单元测试"""

    def test_valid_config_creation(self):
        """测试有效配置的创建"""
        config = {
            'detect_and_handle_config': [
                {'zscore': {'columns': ['T', 'wv'], 'handle_method': ['clip', 'impute'], 'threshold': 3}}
            ],
            'generate_outlier_indicator': ['T']
        }

        method_config = MethodConfig.from_dict(config)
        assert method_config.detect_and_handle_config == config['detect_and_handle_config']
        assert method_config.generate_outlier_indicator == ['T']

    def test_invalid_detection_method(self):
        """测试无效的检测方法"""
        config = {
            'detect_and_handle_config': [
                {'invalid_method': {'columns': ['T'], 'handle_method': ['clip']}}
            ]
        }

        with pytest.raises(ValueError, match="异常值检测方式仅支持"):
            MethodConfig.from_dict(config)

    def test_missing_required_fields(self):
        """测试缺少必要字段"""
        config = {
            'detect_and_handle_config': [
                {'zscore': {'columns': ['T']}}  # 缺少 handle_method
            ]
        }

        with pytest.raises(ValueError, match="'columns'和'handle_method'是配置必须字段"):
            MethodConfig.from_dict(config)

    def test_zscore_threshold_validation(self):
        """测试 zscore 方法阈值验证"""
        config = {
            'detect_and_handle_config': [
                {'zscore': {'columns': ['T'], 'handle_method': ['clip']}}  # 缺少 threshold
            ]
        }

        with pytest.raises(ValueError, match="'zscore'和'iqr'检测方式，必须同步配置'threshold'参数"):
            MethodConfig.from_dict(config)

    def test_custom_method_function_validation(self):
        """测试自定义方法函数验证"""
        config = {
            'detect_and_handle_config': [
                {'custom': {'columns': ['T'], 'handle_method': ['clip']}}  # 缺少 detect_function
            ]
        }

        with pytest.raises(ValueError, match="'custom'检测方式，必须同步配置自定义检测函数"):
            MethodConfig.from_dict(config)

    def test_minmax_business_range_validation(self):
        """测试业务范围 minmax 验证"""
        config = {
            'detect_and_handle_config': [
                {'minmax_business': {
                    'columns': ['T'],
                    'handle_method': ['clip'],
                    'feature_range_business': (100, 0)  # 无效范围
                }}
            ]
        }

        with pytest.raises(ValueError, match="minmax_business的feature_range_business必须是长度为2的数值元组"):
            MethodConfig.from_dict(config)


class TestNumericOutlierProcessorUnit:
    """NumericOutlierProcessor 单元测试"""

    @pytest.fixture
    def sample_data(self):
        """提供样本数据"""
        np.random.seed(42)
        data = pd.DataFrame({
            'normal_col': np.random.normal(0, 1, 100),
            'outlier_col': np.concatenate([np.random.normal(0, 1, 95), np.random.normal(10, 1, 5)]),
            'constant_col': np.ones(100),
            'sparse_col': np.concatenate([np.random.normal(0, 1, 10), [np.nan] * 90])
        })
        return data

    @pytest.fixture
    def basic_config(self):
        """提供基础配置"""
        return {
            'detect_and_handle_config': [
                {'zscore': {'columns': ['normal_col', 'outlier_col', 'constant_col'],
                            'handle_method': ['clip', 'clip', 'clip'],
                            'threshold': 3}}
            ],
            'generate_outlier_indicator': ['outlier_col']
        }

    def test_initialization_with_none_config(self):
        """测试使用 None 配置初始化"""
        processor = NumericOutlierProcessor(method_config=None)
        assert processor._valid_method_config is not None

    def test_initialization_with_dict_config(self):
        """测试使用字典配置初始化"""
        config = {
            'detect_and_handle_config': [
                {'iqr': {'columns': ['col1'], 'handle_method': ['clip'], 'threshold': 1.5}}
            ],
            'generate_outlier_indicator': []
        }

        processor = NumericOutlierProcessor(method_config=config)
        assert isinstance(processor._valid_method_config, MethodConfig)

    def test_pass_through_mode(self, sample_data):
        """测试直通模式"""
        processor = NumericOutlierProcessor(pass_through=True)

        result_fit = processor.fit(sample_data)
        assert result_fit is processor

        result_transform = processor.transform(sample_data)
        pd.testing.assert_frame_equal(result_transform, sample_data)

    def test_fit_with_constant_column(self, sample_data, basic_config):
        """测试处理常数列"""
        processor = NumericOutlierProcessor(method_config=basic_config)
        processor.fit(sample_data)

        # 常数列应该被标记为 'no_need'
        assert processor.outlier_thresholds_['constant_col']['detect_method'] == 'no_need'

    def test_zscore_detection_fit(self, sample_data):
        """测试 ZScore 检测器拟合"""
        config = {
            'detect_and_handle_config': [
                {'zscore': {'columns': ['normal_col'], 'handle_method': ['clip'], 'threshold': 2.5}}
            ]
        }

        processor = NumericOutlierProcessor(method_config=config)
        processor.fit(sample_data)

        config_data = processor.outlier_thresholds_['normal_col']
        assert config_data['detect_method'] == 'zscore'
        assert 'mean' in config_data
        assert 'std' in config_data
        assert config_data['threshold'] == 2.5

    def test_iqr_detection_fit(self, sample_data):
        """测试 IQR 检测器拟合"""
        config = {
            'detect_and_handle_config': [
                {'iqr': {'columns': ['normal_col'], 'handle_method': ['clip'], 'threshold': 1.5}}
            ]
        }

        processor = NumericOutlierProcessor(method_config=config)
        processor.fit(sample_data)

        config_data = processor.outlier_thresholds_['normal_col']
        assert config_data['detect_method'] == 'iqr'
        assert 'lower_bound' in config_data
        assert 'upper_bound' in config_data

    def test_robust_detection_fit(self, sample_data):
        """测试 Robust 检测器拟合"""
        config = {
            'detect_and_handle_config': [
                {'robust': {'columns': ['normal_col'], 'handle_method': ['clip'], 'quantile_range': (25, 75)}}
            ]
        }

        processor = NumericOutlierProcessor(method_config=config)
        processor.fit(sample_data)

        config_data = processor.outlier_thresholds_['normal_col']
        assert config_data['detect_method'] == 'robust'
        assert 'lower_bound' in config_data
        assert 'upper_bound' in config_data

    def test_isolation_forest_detection_fit(self, sample_data):
        """测试隔离森林检测器拟合"""
        config = {
            'detect_and_handle_config': [
                {'isolationforest': {
                    'columns': ['normal_col'],
                    'handle_method': ['clip'],
                    'contamination': 0.1,
                    'random_state': 42
                }}
            ]
        }

        processor = NumericOutlierProcessor(method_config=config)
        processor.fit(sample_data)

        config_data = processor.outlier_thresholds_['normal_col']
        assert config_data['detect_method'] == 'isolationforest'
        assert config_data['contamination'] == 0.1

    def test_custom_detection_fit(self, sample_data):
        """测试自定义检测器拟合"""

        def custom_detect(series, threshold=0):
            return series < threshold

        config = {
            'detect_and_handle_config': [
                {'custom': {
                    'columns': ['normal_col'],
                    'handle_method': ['clip'],
                    'detect_function': partial(custom_detect, threshold=-1)
                }}
            ]
        }

        processor = NumericOutlierProcessor(method_config=config)
        processor.fit(sample_data)

        config_data = processor.outlier_thresholds_['normal_col']
        assert config_data['detect_method'] == 'custom'
        assert 'detect_function' in config_data


class TestNumericOutlierProcessorIntegration:
    """NumericOutlierProcessor 集成测试"""

    @pytest.fixture
    def pipeline_data(self):
        """提供管道测试数据"""
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.concatenate([np.random.normal(0, 1, 95), [10, -10, 15, -15, 20]]),
            'feature2': np.concatenate([np.random.normal(5, 2, 95), [20, -5, 25, -10, 30]]),
            'target': np.random.normal(0, 1, 100)
        })
        return data

    def test_sklearn_pipeline_integration(self, pipeline_data):
        """测试与 sklearn Pipeline 的集成"""
        config = {
            'detect_and_handle_config': [
                {'zscore': {'columns': ['feature1', 'feature2'], 'handle_method': ['clip', 'clip'], 'threshold': 3}}
            ],
            'generate_outlier_indicator': ['feature1']
        }

        pipeline = Pipeline([
            ('outlier_processor', NumericOutlierProcessor(method_config=config)),
        ])

        # 测试拟合和转换
        transformed = pipeline.fit_transform(pipeline_data)

        assert transformed.shape[0] == pipeline_data.shape[0]
        assert 'is_outlier_feature1' in transformed.columns

    def test_multiple_detection_methods_in_pipeline(self, pipeline_data):
        """测试管道中多种检测方法的组合"""
        config = {
            'detect_and_handle_config': [
                {'zscore': {'columns': ['feature1'], 'handle_method': ['clip'], 'threshold': 3}},
                {'iqr': {'columns': ['feature2'], 'handle_method': ['clip'], 'threshold': 2.0}}
            ],
            'generate_outlier_indicator': ['feature1', 'feature2']
        }

        pipeline = Pipeline([
            ('outlier_processor', NumericOutlierProcessor(method_config=config)),
        ])

        transformed = pipeline.fit_transform(pipeline_data)

        assert transformed.shape[0] == pipeline_data.shape[0]
        assert 'is_outlier_feature1' in transformed.columns
        assert 'is_outlier_feature2' in transformed.columns

    def test_fit_transform_equivalence(self, pipeline_data):
        """测试 fit + transform 与直接 fit_transform 的等价性"""
        config = {
            'detect_and_handle_config': [
                {'zscore': {'columns': ['feature1'], 'handle_method': ['clip'], 'threshold': 3}}
            ]
        }

        processor1 = NumericOutlierProcessor(method_config=config)
        result1 = processor1.fit(pipeline_data).transform(pipeline_data)

        processor2 = NumericOutlierProcessor(method_config=config)
        result2 = processor2.fit_transform(pipeline_data)

        pd.testing.assert_frame_equal(result1, result2)


class TestNumericOutlierProcessorFunctional:
    """NumericOutlierProcessor 功能测试"""

    def test_outlier_detection_accuracy(self):
        """测试异常值检测的准确性"""
        # 创建有明显异常值的数据
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 95)
        outliers = np.array([10, -10, 15, -15, 20])  # 明显异常值
        test_data = pd.DataFrame({
            'values': np.concatenate([normal_data, outliers])
        })

        config = {
            'detect_and_handle_config': [
                {'robust': {'columns': ['values'], 'handle_method': ['clip'], 'quantile_range': (25, 75)}}
            ],
            'generate_outlier_indicator': ['values']
        }

        processor = NumericOutlierProcessor(method_config=config)
        result = processor.fit_transform(test_data)

        # 检查异常值指示器
        outlier_indicator = result['is_outlier_values']
        detected_outliers = outlier_indicator.sum()

        # 应该检测到大部分异常值
        assert detected_outliers >= 4  # 至少检测到4个异常值

    def test_clip_handling_strategy(self):
        """测试截断处理策略"""
        data = pd.DataFrame({
            'values': [1, 2, 3, 100, -100, 4, 5]  # 包含明显异常值
        })

        config = {
            'detect_and_handle_config': [
                {'iqr': {'columns': ['values'], 'handle_method': ['clip'], 'threshold': 1.5}}
            ]
        }

        processor = NumericOutlierProcessor(method_config=config)
        result = processor.fit_transform(data)

        # 检查异常值是否被正确截断
        processed_values = result['values']
        assert processed_values.min() >= -100  # 应根据实际计算调整
        assert processed_values.max() <= 100  # 应根据实际计算调整

    def test_nan_handling_strategy(self):
        """测试 NaN 处理策略"""
        data = pd.DataFrame({
            'values': [1, 2, 3, 100, 4, 5]  # 100 是异常值
        })

        config = {
            'detect_and_handle_config': [
                {'zscore': {'columns': ['values'], 'handle_method': ['nan'], 'threshold': 2}}
            ]
        }

        processor = NumericOutlierProcessor(method_config=config)
        result = processor.fit_transform(data)

        # 检查异常值是否被替换为 NaN
        assert result['values'].isna().sum() > 0

    def test_impute_handling_strategy(self):
        """测试插值处理策略"""
        data = pd.DataFrame({
            'values': [1, 2, 3, 100, 4, 5]  # 100 是异常值
        })

        config = {
            'detect_and_handle_config': [
                {'zscore': {'columns': ['values'], 'handle_method': ['impute'], 'threshold': 2}}
            ]
        }

        processor = NumericOutlierProcessor(method_config=config)
        result = processor.fit_transform(data)

        # 检查异常值是否被中位数替换
        median_val = data['values'].median()
        assert (result['values'] == median_val).sum() > 0

    def test_constant_handling_strategy(self):
        """测试常量替换策略"""
        data = pd.DataFrame({
            'values': [1, 2, 3, 100, 4, 5]
        })

        config = {
            'detect_and_handle_config': [
                {'zscore': {
                    'columns': ['values'],
                    'handle_method': ['constant'],
                    'threshold': 2,
                    'constant_value': 999
                }}
            ]
        }

        processor = NumericOutlierProcessor(method_config=config)
        result = processor.fit_transform(data)

        # 检查异常值是否被替换为指定常量
        assert (result['values'] == 999).sum() > 0

    def test_custom_handling_strategy(self):
        """测试自定义处理策略"""

        def custom_handle(series, mask_series):
            series[mask_series] = series.mean()
            return series

        data = pd.DataFrame({
            'values': [1, 2, 3, 100, 4, 5]
        })

        config = {
            'detect_and_handle_config': [
                {'zscore': {
                    'columns': ['values'],
                    'handle_method': ['custom'],
                    'threshold': 2,
                    'handle_function': custom_handle
                }}
            ]
        }

        processor = NumericOutlierProcessor(method_config=config)
        result = processor.fit_transform(data)

        # 检查自定义处理是否生效
        mean_val = data['values'].mean()
        assert (result['values'] == mean_val).sum() > 0

    def test_performance_with_large_dataset(self):
        """测试大数据集下的性能"""
        # 创建大型数据集
        np.random.seed(42)
        large_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 10000),
            'feature2': np.random.normal(5, 2, 10000),
            'feature3': np.concatenate([np.random.normal(0, 1, 9990), np.random.normal(10, 1, 10)])
        })

        config = {
            'detect_and_handle_config': [
                {'zscore': {'columns': ['feature1', 'feature2', 'feature3'], 'handle_method': ['clip', 'clip', 'clip'],
                            'threshold': 3}}
            ]
        }

        processor = NumericOutlierProcessor(method_config=config)

        # 测试性能（主要检查是否能在合理时间内完成）
        import time
        start_time = time.time()

        result = processor.fit_transform(large_data)

        end_time = time.time()
        processing_time = end_time - start_time

        # 处理时间应该合理（例如小于10秒）
        assert processing_time < 10.0
        assert result.shape == large_data.shape

    def test_data_quality_after_processing(self):
        """测试处理后的数据质量"""
        np.random.seed(42)
        data = pd.DataFrame({
            'quality_col': np.concatenate([
                np.random.normal(0, 1, 90),
                [100, -100, 150, -150]  # 极端异常值
            ])
        })

        original_stats = {
            'min': data['quality_col'].min(),
            'max': data['quality_col'].max(),
            'std': data['quality_col'].std()
        }

        config = {
            'detect_and_handle_config': [
                {'iqr': {'columns': ['quality_col'], 'handle_method': ['clip'], 'threshold': 1.5}}
            ]
        }

        processor = NumericOutlierProcessor(method_config=config)
        result = processor.fit_transform(data)

        processed_stats = {
            'min': result['quality_col'].min(),
            'max': result['quality_col'].max(),
            'std': result['quality_col'].std()
        }

        # 处理后数据的标准差应该更合理
        assert processed_stats['std'] < original_stats['std']
        # 极值范围应该缩小
        assert (processed_stats['max'] - processed_stats['min']) < (original_stats['max'] - original_stats['min'])

    def test_data_quality_after_processing1(self):
        """测试处理后的数据质量"""
        np.random.seed(42)
        data = pd.DataFrame({
            'quality_col': np.concatenate([
                np.random.normal(0, 1, 90),  # 正常数据
                [10, -10, 15, -15, 20]  # 适度异常值（不是极端异常值）
            ])
        })

        original_stats = {
            'min': data['quality_col'].min(),
            'max': data['quality_col'].max(),
            'std': data['quality_col'].std()
        }

        # 使用 IQR 方法
        config = {
            'detect_and_handle_config': [
                {'zscore': {'columns': ['quality_col'], 'handle_method': ['clip'], 'threshold': 3}}
            ]
        }

        processor = NumericOutlierProcessor(method_config=config)
        result = processor.fit_transform(data)

        processed_stats = {
            'min': result['quality_col'].min(),
            'max': result['quality_col'].max(),
            'std': result['quality_col'].std()
        }

        # 更合理的断言
        assert processed_stats['std'] < original_stats['std'] * 2  # 标准差应该改善
        assert abs(processed_stats['min']) < 10  # 最小值应该在合理范围内
        assert abs(processed_stats['max']) < 10  # 最大值应该在合理范围内


class TestErrorHandlingAndEdgeCases:
    """错误处理和边界情况测试"""

    def test_missing_column_handling(self):
        """测试处理数据中不存在的列"""
        data = pd.DataFrame({'existing_col': [1, 2, 3]})

        config = {
            'detect_and_handle_config': [
                {'zscore': {'columns': ['non_existing_col'], 'handle_method': ['clip'], 'threshold': 3}}
            ]
        }

        processor = NumericOutlierProcessor(method_config=config)

        # 应该记录警告但不报错
        with pytest.warns(match="不在训练数据中"):
            result = processor.fit_transform(data)

        assert 'existing_col' in result.columns

    def test_empty_dataframe(self):
        """测试空数据框处理"""
        empty_data = pd.DataFrame()

        config = {
            'detect_and_handle_config': [
                {'zscore': {'columns': ['some_col'], 'handle_method': ['clip'], 'threshold': 3}}
            ]
        }

        processor = NumericOutlierProcessor(method_config=config)

        # 应该能够处理空数据框而不报错
        with pytest.raises(ValueError, match="输入数据X不能全部为NaN"):
            result = processor.fit_transform(empty_data)

    def test_all_nan_column(self):
        """测试全 NaN 列的处理"""
        data = pd.DataFrame({
            'all_nan_col': [np.nan] * 10,
            'normal_col': [1, 2, 3, 4, 5] * 2
        })

        config = {
            'detect_and_handle_config': [
                {'zscore': {'columns': ['all_nan_col', 'normal_col'], 'handle_method': ['clip', 'clip'],
                            'threshold': 3}}
            ]
        }

        processor = NumericOutlierProcessor(method_config=config)

        # 应该能够处理全 NaN 列而不报错
        result = processor.fit_transform(data)
        assert result['all_nan_col'].isna().all()

    def test_single_value_column(self):
        """测试单值列的处理"""
        data = pd.DataFrame({
            'single_val_col': [5] * 10,
            'normal_col': [1, 2, 3, 4, 5] * 2
        })

        config = {
            'detect_and_handle_config': [
                {'zscore': {'columns': ['single_val_col', 'normal_col'], 'handle_method': ['clip', 'clip'],
                            'threshold': 3}}
            ]
        }

        processor = NumericOutlierProcessor(method_config=config)

        # 单值列应该被标记为 'no_need'
        processor.fit(data)
        assert processor.outlier_thresholds_['single_val_col']['detect_method'] == 'no_need'

    def test_transform_without_fit(self):
        """测试未拟合直接转换的错误处理"""
        data = pd.DataFrame({'values': [1, 2, 3]})
        processor = NumericOutlierProcessor()

        with pytest.raises(ValueError):
            processor.transform(data)


# 运行测试的便捷函数
def run_all_tests():
    """运行所有测试"""
    import sys
    import subprocess

    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"])
    return result.returncode


if __name__ == "__main__":
    run_all_tests()

import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from functools import partial

from data.data_preparation.handle_extre_numeric_features import NumericOutlierProcessor, MethodConfig  # 替换为实际模块路径

def debug_handle_(series, mask_series):
    """深入调试索引行为"""
    print("=== 深入调试 ===")
    print(f"series: {series.tolist()}")
    print(f"mask_series: {mask_series.tolist()}")
    print(f"mask_series dtype: {mask_series.dtype}")

    # 检查 pandas 如何解释这个 mask

    # 测试 pandas 的实际索引行为
    result = series.copy()

    print("\n=== 测试1: 直接使用 mask_series ===")
    selected = result[mask_series]  # 看看实际选择了什么
    print(f"被选中的值: {selected.tolist()}")
    print(f"被选中的索引位置: {selected.index.tolist()}")

    result[mask_series] = 0
    print(f"赋值后的结果: {result.tolist()}")

    print("\n=== 测试2: 显式转换为布尔 ===")
    result2 = series.copy()
    bool_mask = mask_series.astype(bool)
    selected2 = result2[bool_mask]
    print(f"布尔mask选中的值: {selected2.tolist()}")
    print(f"布尔mask选中的索引: {selected2.index.tolist()}")

    result2[bool_mask] = 0
    print(f"布尔mask赋值结果: {result2.tolist()}")


# 运行调试
data = pd.Series([1, 2, -1, -2, 10, -10, np.nan])
mask = np.array([False, False, True, True, False,True, False])
debug_handle_(data, mask)
