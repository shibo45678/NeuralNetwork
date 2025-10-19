# test_unified_scaler_unit.py
import pytest
import pandas as pd
import numpy as np
from data.feature_engineer.scalers import UnifiedFeatureScaler, SmartScalerSelector,AlgorithmAwareScalerSelector




class TestUnifiedFeatureScaler:

    def test_initialization(self):
        """测试初始化"""
        scaler = UnifiedFeatureScaler()
        assert scaler.algorithm == 'neural_network'
        assert scaler.scaling_config == {}

        # 测试带配置初始化
        method_config = {
            'transformers': [('minmax', {'columns': ['test_col']})],
            'skip_scale': ['skip_col']
        }
        scaler_with_config = UnifiedFeatureScaler(method_config=method_config)
        assert scaler_with_config.transformers is not None

    def test_fit_transform_basic(self):
        """测试基本的 fit_transform"""
        X = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, 30, 40, 50]
        })

        scaler = UnifiedFeatureScaler()
        X_scaled = scaler.fit_transform(X,y=None)

        assert X_scaled.shape == X.shape
        assert 'col1' in scaler.scaling_config
        assert 'col2' in scaler.scaling_config

    def test_constant_column(self):
        """测试常数列处理"""
        X = pd.DataFrame({
            'constant_col': [5, 5, 5, 5, 5],
            'normal_col': [1, 2, 3, 4, 5]
        })

        scaler = UnifiedFeatureScaler()
        X_scaled = scaler.fit_transform(X)

        # 检查常数列配置
        constant_config = scaler.scaling_config['constant_col']
        assert 'constant' in constant_config['method']

    def test_inverse_transform(self):
        """测试逆变换"""
        X = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, 30, 40, 50]
        })

        scaler = UnifiedFeatureScaler()
        X_scaled = scaler.fit_transform(X)
        X_restored = scaler.inverse_transform(X_scaled)

        # 检查恢复的准确性
        pd.testing.assert_frame_equal(X, X_restored, check_dtype=False, rtol=1e-10)

    def test_skip_scale_config(self):
        """测试跳过标准化配置"""
        method_config = {
            'skip_scale': ['skip_col']
        }

        X = pd.DataFrame({
            'skip_col': [1, 2, 3, 4, 5],
            'scale_col': [10, 20, 30, 40, 50]
        })

        scaler = UnifiedFeatureScaler(method_config=method_config)
        X_scaled = scaler.fit_transform(X)

        # 检查跳过的列配置
        skip_config = scaler.scaling_config['skip_col']
        assert skip_config['method'] == 'skip_scale'

    def test_specified_transformers(self):
        """测试指定的转换器配置"""
        method_config = {
            'transformers': [
                ('minmax', {'columns': ['minmax_col'], 'feature_range': (0, 1)}),
                ('standard', {'columns': ['standard_col']})
            ]
        }

        X = pd.DataFrame({
            'minmax_col': [1, 2, 3, 4, 5],
            'standard_col': [10, 20, 30, 40, 50]
        })

        scaler = UnifiedFeatureScaler(method_config=method_config)
        scaler.fit(X)

        # 检查配置是否正确应用
        minmax_config = scaler.scaling_config['minmax_col']
        standard_config = scaler.scaling_config['standard_col']

        assert 'minmax' in minmax_config['method']
        assert 'standard' in standard_config['method']


class TestSmartScalerSelector:

    def test_analyze_feature(self):
        """测试特征分析"""
        selector = SmartScalerSelector()

        # 测试正态分布数据
        normal_data = pd.Series([1, 2, 3, 4, 5])
        stats_info = selector.analyze_feature(normal_data)

        assert 'n_samples' in stats_info
        assert 'mean' in stats_info
        assert 'std' in stats_info
        assert not stats_info['is_constant']

    def test_analyze_feature_constant(self):
        """测试常数列分析"""
        selector = SmartScalerSelector()

        constant_data = pd.Series([5, 5, 5, 5, 5])
        stats_info = selector.analyze_feature(constant_data)

        assert stats_info['is_constant'] == True
        assert stats_info['std'] < 1e-8

    def test_analyze_feature_small_sample(self):
        """测试小样本数据分析"""
        selector = SmartScalerSelector()

        small_data = pd.Series([1, 2])  # 只有2个样本
        stats_info = selector.analyze_feature(small_data)

        assert stats_info.get('recommendation') == 'skip'

    def test_recommend_scaler(self):
        """测试标准化方法推荐"""
        selector = SmartScalerSelector()

        # 测试常数列
        constant_stats = {'is_constant': True, 'mean': 5.0}
        method, reason = selector.recommend_scaler(constant_stats, 'minmax')
        assert 'constant' in method

        # 测试异常值情况
        outlier_stats = {
            'is_constant': False,
            'outlier_ratio_z': 0.1,
            'outlier_ratio_iqr': 0.1,
            'skewness': 0.5
        }
        method, reason = selector.recommend_scaler(outlier_stats)
        assert method == 'robust'

        # 测试偏态分布
        skewed_stats = {
            'is_constant': False,
            'outlier_ratio_z': 0.01,
            'outlier_ratio_iqr': 0.01,
            'skewness': 3.0,  # 高偏度
            'range': 100
        }
        method, reason = selector.recommend_scaler(skewed_stats)
        assert method == 'minmax'

    def test_process_dataframe(self):
        """测试处理整个DataFrame"""
        selector = SmartScalerSelector()

        X = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [5, 5, 5, 5, 5],  # 常数列
            'col3': [1, 2, 100, 101, 102]  # 有异常值
        })

        selector.process(X)
        recommendations = selector.get_recommendations()

        assert 'col1' in recommendations
        assert 'col2' in recommendations
        assert 'col3' in recommendations
        assert recommendations['col2']['scaler'] == 'skip'


class TestAlgorithmAwareScalerSelector:

    def test_initialization(self):
        """测试初始化"""
        selector = AlgorithmAwareScalerSelector()
        assert hasattr(selector, 'algorithm_requirements')
        assert 'neural_network' in selector.algorithm_requirements

    def test_recommend_for_algorithm_known(self):
        """测试已知算法的推荐"""
        selector = AlgorithmAwareScalerSelector()

        # 测试神经网络算法
        feature_stats = {
            'scaler': 'standard',  # 假设智能选择器推荐了standard
            'outlier_ratio_iqr': 0.01,
            'skewness': 0.5
        }

        scaler_type, reason = selector.recommend_for_algorithm('neural_network', feature_stats)
        assert scaler_type in ['standard', 'minmax']
        assert '神经网络' in reason or 'neural_network' in reason

    def test_recommend_for_algorithm_unknown(self):
        """测试未知算法的推荐"""
        selector = AlgorithmAwareScalerSelector()

        feature_stats = {'scaler': 'standard'}
        scaler_type, reason = selector.recommend_for_algorithm('unknown_algorithm', feature_stats)

        assert scaler_type == 'standard'
        assert '不在已设置的算法内' in reason

    def test_recommend_with_outliers(self):
        """测试有异常值时的推荐"""
        selector = AlgorithmAwareScalerSelector()

        # 有异常值的情况
        outlier_stats = {
            'scaler': 'standard',
            'outlier_ratio_iqr': 0.1,  # 10%异常值
            'skewness': 0.5
        }

        # 测试SVM算法（优先robust）
        scaler_type, reason = selector.recommend_for_algorithm('svm', outlier_stats)
        assert scaler_type == 'robust'
        assert '异常值' in reason

    def test_recommend_with_skewness(self):
        """测试高偏度时的推荐"""
        selector = AlgorithmAwareScalerSelector()

        # 高偏度的情况
        skewed_stats = {
            'scaler': 'standard',
            'outlier_ratio_iqr': 0.01,
            'skewness': 3.0  # 高偏度
        }

        # 测试随机森林算法
        scaler_type, reason = selector.recommend_for_algorithm('random_forest', skewed_stats)
        assert scaler_type == 'minmax'
        assert '分布偏斜' in reason

    def test_all_supported_algorithms(self):
        """测试所有支持的算法"""
        selector = AlgorithmAwareScalerSelector()

        test_cases = [
            ('neural_network', ['standard', 'minmax']),
            ('cnn', ['standard', 'minmax']),
            ('knn', ['standard', 'minmax']),
            ('svm', ['standard', 'robust']),
            ('linear_regression', ['standard', 'robust']),
            ('random_forest', ['minmax', 'standard']),
            ('xgboost', ['minmax', 'standard']),
        ]

        base_stats = {'scaler': 'standard', 'outlier_ratio_iqr': 0.01, 'skewness': 0.5}

        for algorithm, expected_priorities in test_cases:
            scaler_type, reason = selector.recommend_for_algorithm(algorithm, base_stats)
            assert scaler_type in expected_priorities, f"{algorithm} 推荐了意外的缩放器: {scaler_type}"

    def test_skip_recommendation(self):
        """测试跳过推荐的情况"""
        selector = AlgorithmAwareScalerSelector()

        skip_stats = {'scaler': 'skip'}
        scaler_type, reason = selector.recommend_for_algorithm('neural_network', skip_stats)

        # 当智能选择器建议跳过时，算法感知选择器应该尊重这个决定
        assert scaler_type is None or '数据不足' in reason


class TestIntegration:

    def test_full_integration(self):
        """测试完整集成流程"""
        # 创建测试数据
        X = pd.DataFrame({
            'normal_col': np.random.normal(0, 1, 100),
            'skewed_col': np.random.exponential(2, 100),
            'outlier_col': np.concatenate([np.random.normal(0, 1, 95), np.random.normal(10, 1, 5)]),
            'constant_col': np.ones(100)
        })

        # 使用算法感知的标准化器
        scaler = UnifiedFeatureScaler(algorithm='svm')
        X_scaled = scaler.fit_transform(X)

        # 验证结果
        assert X_scaled.shape == X.shape
        assert len(scaler.scaling_config) == 4

        # 检查报告
        report = scaler.get_scaling_report()
        assert report['algorithm'] == 'svm'
        assert report['summary']['total_features'] == 4

    def test_pipeline_compatibility(self):
        """测试sklearn pipeline兼容性"""
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import RandomForestClassifier

        # 创建分类数据
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])

        # 创建pipeline
        pipeline = Pipeline([
            ('scaler', UnifiedFeatureScaler(algorithm='random_forest')),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])

        # 测试pipeline工作
        pipeline.fit(X_df, y)
        predictions = pipeline.predict(X_df)

        assert len(predictions) == len(y)
        assert hasattr(pipeline.named_steps['scaler'], 'scaling_config')


def make_classification(n_samples=100, n_features=4, random_state=42):
    """创建分类数据的辅助函数"""
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    return X, y


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])