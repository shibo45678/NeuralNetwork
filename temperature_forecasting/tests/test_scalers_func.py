from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from data.feature_engineer.scalers import (UnifiedFeatureScaler, SmartScalerSelector, StatisticsCalculation,
                                           StandardizationDecisionAlgorithm, ScalerType, ConfigParser,
                                            MethodConfig,ScalerFactory)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
class TestUnifiedFeatureScalerIntegration:
    """UnifiedFeatureScaler 集成测试"""

    def test_in_sklearn_pipeline(self):
        """测试在 sklearn Pipeline 中的集成"""
        # 创建管道
        pipeline = Pipeline([
            ('scaler', UnifiedFeatureScaler(algorithm='random_forest')),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])

        # 生成样本数据
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        X_df = pd.DataFrame(X, columns=['col1', 'col2', 'col3', 'col4'])

        # 测试管道运行
        pipeline.fit(X_df, y)
        predictions = pipeline.predict(X_df)

        assert len(predictions) == len(y)
        assert hasattr(pipeline.named_steps['scaler'], 'scaling_config')

    def test_with_real_algorithm_compatibility(self):
        """测试与真实算法的兼容性"""
        algorithms = ['random_forest', 'xgboost', 'neural_network']

        for algo in algorithms:
            scaler = UnifiedFeatureScaler(algorithm=algo)
            X = pd.DataFrame({
                'feature1': np.random.normal(0, 1, 50),
                'feature2': np.random.uniform(0, 100, 50)
            })

            # 应该能正常拟合和转换
            X_scaled = scaler.fit_transform(X)
            assert X_scaled.shape == X.shape
            assert not np.any(np.isnan(X_scaled))

    def test_data_flow_through_system(self):
        """测试数据流通过整个系统"""
        config_dict = {
            'transformers': [
                {'minmax': {'columns': ['feature1'], 'feature_range': (0, 1)}},
                {'standard': {'columns': ['feature2']}}
            ],
            'skip_scale': ['feature3']
        }

        scaler = UnifiedFeatureScaler(
            method_config=config_dict,
            algorithm='neural_network'
        )

        X = pd.DataFrame({
            'feature1': np.random.uniform(0, 100, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.exponential(2, 100),
            'feature4': np.random.normal(10, 5, 100)  # 智能推荐
        })

        # 测试完整的数据流
        X_scaled = scaler.fit_transform(X)

        # 验证结果
        assert X_scaled.shape == X.shape
        report = scaler.get_scaling_report()
        assert 'summary' in report
        assert report['summary']['total_features'] == 4

    def test_inverse_transform_integration(self):
        """测试逆转换集成"""
        scaler = UnifiedFeatureScaler(algorithm='neural_network')
        original_data = pd.DataFrame({
            'col1': np.random.normal(0, 1, 50),
            'col2': np.random.uniform(0, 100, 50)
        })

        # 正向转换
        scaled_data = scaler.fit_transform(original_data)

        # 逆向转换
        restored_data = scaler.inverse_transform(scaled_data)

        # 验证形状一致
        assert restored_data.shape == original_data.shape
        # 数值应该近似（由于浮点精度）
        np.testing.assert_allclose(restored_data.values, original_data.values, rtol=1e-10)