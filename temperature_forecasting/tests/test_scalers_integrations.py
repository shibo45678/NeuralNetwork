import tempfile

from data.feature_engineering.scalers import (UnifiedFeatureScaler)
import pandas as pd
import numpy as np


class TestUnifiedFeatureScalerFunctional:
    """UnifiedFeatureScaler 功能测试"""

    def test_business_requirements_satisfaction(self):
        """测试业务需求是否满足"""
        # 测试场景：用户需要混合标准化策略
        config_dict = {
            'transformers': [
                {'minmax': {'columns': ['temperature', 'pressure'], 'feature_range': (0, 1)}},
                {'standard': {'columns': ['velocity', 'acceleration']}},
                {'robust': {'columns': ['outlier_sensor'], 'quantile_range': (10, 90)}}
            ],
            'skip_scale': ['id', 'timestamp']  # 元数据列跳过
        }

        scaler = UnifiedFeatureScaler(method_config=config_dict)

        # 模拟真实传感器数据
        np.random.seed(42)
        X = pd.DataFrame({
            'temperature': np.random.uniform(-10, 50, 100),
            'pressure': np.random.uniform(900, 1100, 100),
            'velocity': np.random.normal(0, 10, 100),
            'acceleration': np.random.normal(0, 5, 100),
            'outlier_sensor': np.concatenate([
                np.random.normal(0, 1, 95),
                np.random.normal(20, 5, 5)  # 异常值
            ]),
            'id': range(100),
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='H')
        })

        # 验证功能正常
        X_scaled = scaler.fit_transform(X,y=None)
        report = scaler.get_scaling_report()

        # 业务需求验证
        assert X_scaled.shape[1] == X.shape[1]  # 保持所有特征
        assert 'minmax' in str(report['scaling_config']['temperature']['method'])
        assert 'standard' in str(report['scaling_config']['velocity']['method'])
        assert 'user_skip' in str(report['scaling_config']['id']['method'])

    def test_user_scenarios_work_correctly(self):
        """测试用户场景正常工作"""
        scenarios = [
            {
                'name': '完全自动模式',
                'config': None,
                'data': pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]}),
                'expected_columns': 2
            },
            {
                'name': '指定跳过模式',
                'config': {'skip_scale': ['col1']},
                'data': pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]}),
                'expected_user_skip': True
            },
            {
                'name': '混合配置模式',
                'config': {
                    'transformers': [
                        {'minmax': {'columns': ['col1']}},
                        {'standard': {'columns': ['col2']}}
                    ]
                },
                'data': pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]}),
                'expected_methods': ['minmax', 'standard']
            }
        ]

        for scenario in scenarios:
            scaler = UnifiedFeatureScaler(
                method_config=scenario['config'],
                algorithm='random_forest'
            )

            X_scaled = scaler.fit_transform(scenario['data'])
            report = scaler.get_scaling_report()

            # 验证场景特定期望
            if 'expected_columns' in scenario:
                assert X_scaled.shape[1] == scenario['expected_columns']

            if 'expected_user_skip' in scenario:
                assert 'user_skip' in str(report['scaling_config']['col1']['method'])

    def test_performance_with_large_data(self):
        """测试大数据量下的性能"""
        # 生成较大数据集
        n_samples = 10000
        n_features = 50

        X_large = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )

        scaler = UnifiedFeatureScaler(algorithm='neural_network')

        # 测试拟合性能
        import time
        start_time = time.time()
        X_scaled = scaler.fit_transform(X_large)
        end_time = time.time()

        # 验证性能在合理范围内（可根据实际需求调整阈值）
        fitting_time = end_time - start_time
        assert fitting_time < 30.0  # 30秒内完成10000×50的数据处理

        # 验证数据完整性
        assert X_scaled.shape == X_large.shape
        assert not np.any(np.isnan(X_scaled))

    def test_data_quality_handling(self):
        """测试数据质量处理"""
        problematic_data = pd.DataFrame({
            'all_nan': [np.nan, np.nan, np.nan, np.nan],
            'all_constant': [1, 1, 1, 1],
            'nearly_constant': [1, 1, 1, 2],
            'with_inf': [1, 2, np.inf, 4],
            'with_neg_inf': [1, 2, -np.inf, 4],
            'normal': [1, 2, 3, 4]
        })

        scaler = UnifiedFeatureScaler(algorithm='neural_network')

        # 应该能处理有问题的数据而不崩溃
        X_scaled = scaler.fit_transform(problematic_data)
        report = scaler.get_scaling_report()

        # 验证各种数据质量问题的处理
        assert 'standard_skip' in str(report['scaling_config']['all_nan']['method'])
        assert 'cons_skip' in str(report['scaling_config']['all_constant']['method'])

    def test_pass_through_functionality(self):
        """测试 pass_through 功能"""
        # 场景1：用户显式设置 pass_through=True
        scaler_explicit = UnifiedFeatureScaler(pass_through=True)
        X = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})

        X_transformed = scaler_explicit.fit_transform(X)
        pd.testing.assert_frame_equal(X, X_transformed)

        # 场景2：算法计算出
        scaler_implicit = UnifiedFeatureScaler(algorithm='naive_bayes')
        X_transformed_implicit = scaler_implicit.fit_transform(X)

        # 对于 naive_bayes，应该检测到不需要标准化
        report = scaler_implicit.get_scaling_report()
        assert report.get('skip_all', False) is True

    def test_method_config_same_method_different_params(self):
        """测试同一种方法中不同参数配置的检测"""
        config_dict = {
            'transformers': [
                {'minmax': {'columns': ['T', 'S'], 'feature_range': (0, 1)}},  # 相同方法，相同配置
                {'minmax': {'columns': ['W'], 'feature_range': (-1, 1)}},  # 相同方法，不同配置
                {'minmax': {'columns': ['V'], 'feature_range': (0, 1)}},  # 相同方法，相同配置（应该去重）
                {'standard': {'columns': ['P', 'Q']}}
            ]
        }

        scaler = UnifiedFeatureScaler(method_config=config_dict)
        X = pd.DataFrame({
            'T': [1, 2, 3], 'S': [4, 5, 6], 'W': [7, 8, 9],
            'V': [10, 11, 12], 'P': [13, 14, 15], 'Q': [16, 17, 18]
        })

        # 处理配置
        scaler._process_params_config(X)

        # 验证配置解析正确
        assert scaler.params_info['T']['params_method'] == 'minmax'
        assert scaler.params_info['W']['params_method'] == 'minmax'
        assert scaler.params_info['W']['params_config']['feature_range'] == (-1, 1)
        assert scaler.params_info['V']['params_method'] == 'minmax'
        assert scaler.params_info['V']['params_config']['feature_range'] == (0, 1)

    def test_edge_cases_and_boundary_conditions(self):
        """测试边界条件和边缘情况"""
        edge_cases = [
            {
                'name': '单样本数据',
                'data': pd.DataFrame({'col1': [1]}),
                'algorithm': 'random_forest'
            },
            {
                'name': '单特征数据',
                'data': pd.DataFrame({'col1': [1, 2, 3, 4, 5]}),
                'algorithm': 'neural_network'
            },
            {
                'name': '全零数据',
                'data': pd.DataFrame({'col1': [0, 0, 0, 0]}),
                'algorithm': 'kmeans'
            },
            {
                'name': '极大值数据',
                'data': pd.DataFrame({'col1': [1e10, 2e10, 3e10]}),
                'algorithm': 'cnn'
            },
            {
                'name': '极小数数据',
                'data': pd.DataFrame({'col1': [1e-10, 2e-10, 3e-10]}),
                'algorithm': 'naive_bayes'
            }
        ]

        for case in edge_cases:
            scaler = UnifiedFeatureScaler(algorithm=case['algorithm'])

            # 应该能处理而不崩溃
            try:
                X_scaled = scaler.fit_transform(case['data'],y=None)
                assert X_scaled.shape == case['data'].shape
            except Exception as e:
                # 某些边界情况允许失败，但应该给出有意义的错误信息
                # assert '尚未配置' in str(e)  # 错误信息应该有意义
                assert '样本' in str(e) or '标准' in str(e)


    def test_serialization_compatibility(self):
        """测试序列化兼容性（与 pickle 等）"""
        import pickle

        # 创建配置完整的 scaler
        config_dict = {
            'transformers': [
                {'minmax': {'columns': ['feature1']}},
                {'standard': {'columns': ['feature2']}}
            ]
        }

        original_scaler = UnifiedFeatureScaler(
            method_config=config_dict,
            algorithm='neural_network'
        )

        X = pd.DataFrame({
            'feature1': np.random.uniform(0, 100, 50),
            'feature2': np.random.normal(0, 1, 50)
        })

        # 拟合数据
        original_scaler.fit(X)

        # 序列化和反序列化
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            pickle.dump(original_scaler, f)
            temp_path = f.name

        with open(temp_path, 'rb') as f:
            restored_scaler = pickle.load(f)

        # 验证反序列化后的对象功能正常
        X_transformed = restored_scaler.transform(X)
        assert X_transformed.shape == X.shape

        # 清理临时文件
        import os
        os.unlink(temp_path)