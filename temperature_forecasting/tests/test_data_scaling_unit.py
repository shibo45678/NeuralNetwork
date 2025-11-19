from data.feature_engineering.feature_scaling import (UnifiedFeatureScaler, SmartScalerSelector, StatisticsCalculation,
                                              ConfigParser,
                                              MethodConfig, ScalerFactory)

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler



class TestFillConfig:
    """测试 FillConfig 配置类"""

    # def test_fill_config_creation(self):
    #     """测试 FillConfig 正常创建"""
    #     config = FillConfig(
    #         columns=['col1', 'col2'],
    #         feature_range=(0, 1),
    #         with_mean=True
    #     )
    #     assert config.columns == ['col1', 'col2']
    #     assert config.feature_range == (0, 1)
    #     assert config.with_mean is True


    def test_valid_minmax_ranges(self):
        """测试各种合法的minmax范围"""
        valid_ranges = [
            (0, 1),  # 默认范围
            (-1, 1),  # 对称范围
            (0, 100),  # 放大范围
            (-5, 5),  # 自定义对称范围
            (0.1, 0.9),  # 浮点数范围
            (-10, 0)  # 负值范围
        ]

        for feature_range in valid_ranges:
            config = MethodConfig(
                transformers=[
                    {'minmax': {
                        'columns': ['col1'],
                        'feature_range': feature_range
                    }}
                ]
            )
            # 应该都不会报错
            fill_config = config.transformers[0]['minmax']
            assert fill_config['feature_range'] == feature_range

    def test_invalid_minmax_ranges(self):
        """测试非法的minmax范围"""
        invalid_ranges = [
            (1, 0),  # 第一个值大于第二个值
            (0,),  # 长度不为2
            (0, 1, 2),  # 长度不为2
            ('a', 'b'),  # 非数值类型
            (0, '1'),  # 混合类型
        ]

        for feature_range in invalid_ranges:
            with pytest.raises(ValueError, match="feature_range"):
                MethodConfig(
                    transformers=[
                        {'minmax': {
                            'columns': ['col1'],
                            'feature_range': feature_range
                        }}
                    ]
                )


class TestMethodConfig:
    """测试 MethodConfig 配置类"""

    def test_method_config_from_dict(self):
        """测试从字典创建 MethodConfig"""
        config_dict = {
            'transformers': [
                {'minmax': {'columns': ['T', 'S'], 'feature_range': (0, 1)}},
                {'standard': {'columns': ['P', 'Q']}}
            ],
            'skip_scale': ['col1', 'col2']
        }

        config = MethodConfig.from_dict(config_dict)
        assert len(config.transformers) == 2
        assert config.skip_scale == ['col1', 'col2']

    def test_method_config_validation_transformers_skip_overlap(self):
        """测试 transformers 和 skip_scale 重叠验证"""
        config_dict = {
            'transformers': [
                {'minmax': {'columns': ['col1', 'col2']}}
            ],
            'skip_scale': ['col1']  # 重叠列
        }

        with pytest.raises(ValueError, match="transformers.*skip_scale.*有重叠列"):
            MethodConfig.from_dict(config_dict)

    def test_method_config_validation_skip_internal_overlap(self):
        """测试 skip_scale 内部重复验证"""
        config_dict = {
            'skip_scale': ['col1', 'col1']  # 内部重复
        }

        with pytest.raises(ValueError, match="skip_scale配置中存在相同列"):
            MethodConfig.from_dict(config_dict)

    def test_method_config_validation_transformers_internal_overlap1(self):
        """测试 transformers 内部列重叠验证"""
        config_dict = {
            'transformers': [
                {'minmax': {'columns': ['col1', 'col2']}},
                {'standard': {'columns': ['col1']}}  # 列重复
            ]
        }

        with pytest.raises(ValueError, match="在多个transfomers方法中重复指定"):
            MethodConfig.from_dict(config_dict)

    def test_method_config_validation_transformers_internal_overlap2(self):
        """测试 transformers 内部列重叠验证"""
        config_dict = {
            'transformers': [
                {'minmax': {'columns': ['col1', 'col2'],'feature_range':(0,1)}},
                {'minmax': {'columns': ['col2'], 'feature_range': (-1, 1)}},
            ]
        }

        with pytest.raises(ValueError, match="在多个transfomers方法中重复指定"):
            MethodConfig.from_dict(config_dict)


class TestScalerFactory:
    """测试 ScalerFactory 工厂类"""

    def test_create_minmax_scaler(self):
        """测试创建 MinMaxScaler"""
        # 测试基本创建
        scaler = ScalerFactory.create_scaler('minmax')
        assert isinstance(scaler, MinMaxScaler)
        assert scaler.feature_range == (0, 1)

        # 测试自定义配置
        scaler = ScalerFactory.create_scaler('minmax', {'feature_range': (-1, 1)})
        assert isinstance(scaler, MinMaxScaler)
        assert scaler.feature_range == (-1, 1)

    def test_create_standard_scaler(self):
        """测试创建 StandardScaler"""
        scaler = ScalerFactory.create_scaler('standard', {'with_std': False})
        assert isinstance(scaler, StandardScaler)
        assert scaler.with_std is False

    def test_create_robust_scaler(self):
        """测试创建 RobustScaler"""
        scaler = ScalerFactory.create_scaler('robust', {'quantile_range': (10, 90)})
        assert isinstance(scaler, RobustScaler)
        assert scaler.quantile_range == (10, 90)

    def test_create_unsupported_scaler(self):
        """测试创建不支持的 scaler 类型"""
        with pytest.raises(ValueError, match="不支持的scaler类型"):
            ScalerFactory.create_scaler('unsupported')


class TestConfigParser:
    """测试 ConfigParser 配置解析器"""

    def test_parse_transformers_basic(self):
        """测试基本 transformers 解析"""
        transformers = [
            {'minmax': {'columns': ['col1', 'col2'], 'feature_range': (0, 1)}},
            {'standard': {'columns': ['col3']}}
        ]
        X_columns = ['col1', 'col2', 'col3', 'col4']

        result = ConfigParser.parse_transformers(transformers, X_columns)

        assert 'col1' in result
        assert 'col2' in result
        assert 'col3' in result
        assert 'col4' not in result  # 不在 transformers 中
        assert result['col1']['params_method'] == 'minmax'
        assert result['col1']['params_config']['feature_range'] == (0, 1)
        assert result['col2']['params_config']['feature_range'] == (0, 1)

    def test_parse_transformers_missing_columns(self):
        """测试解析 transformers 中不存在的列"""
        transformers = [
            {'minmax': {'columns': ['nonexistent']}}
        ]
        X_columns = ['col1', 'col2']

        result = ConfigParser.parse_transformers(transformers, X_columns)
        assert 'nonexistent' not in result  # 应该被过滤掉


class TestUnifiedFeatureScalerUnit:
    """UnifiedFeatureScaler 单元测试"""

    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        np.random.seed(42)
        return pd.DataFrame({
            'normal_col': np.random.normal(0, 1, 100),
            'constant_col': np.ones(100),
            'skewed_col': np.random.exponential(2, 100),
            'outlier_col': np.concatenate([np.random.normal(0, 1, 95), np.random.normal(10, 1, 5)])
        })

    def test_initialize_method_config_dict(self):
        """测试从字典初始化 method_config"""
        config_dict = {
            'transformers': [
                {'minmax': {'columns': ['col1']}}
            ]
        }
        scaler = UnifiedFeatureScaler(method_config=config_dict)
        assert isinstance(scaler.method_config, MethodConfig)

    def test_initialize_method_config_object(self):
        """测试从 MethodConfig 对象初始化"""
        config = MethodConfig(skip_scale=['col1'])
        scaler = UnifiedFeatureScaler(method_config=config)
        assert scaler.method_config == config

    def test_pass_through_mode(self):
        """测试 pass_through 模式"""
        scaler = UnifiedFeatureScaler(pass_through=True)
        X = pd.DataFrame({'col1': [1, 2, 3]})

        X_transformed = scaler.fit_transform(X,y=None)
        pd.testing.assert_frame_equal(X, X_transformed)

    def test_process_params_config(self):
        """测试处理参数配置"""
        config_dict = {
            'transformers': [
                {'minmax': {'columns': ['col1']}},
                {'standard': {'columns': ['col2']}}
            ],
            'skip_scale': ['col3']
        }
        scaler = UnifiedFeatureScaler(method_config=config_dict)
        X = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9],
            'col4': [10, 11, 12]  # 未配置的列
        })

        # 手动调用内部方法进行测试
        scaler._process_params_config(X)

        assert 'col1' in scaler.params_info_
        assert 'col2' in scaler.params_info_
        assert 'col3' in scaler.params_info_
        assert scaler.params_info_['col3'].get('params_method') == 'user_skip'
        assert 'col4' not in scaler.params_info_


class TestSmartScalerSelector:
    """测试 SmartScalerSelector"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        return pd.DataFrame({
            'normal': np.random.normal(0, 1, 50),
            'constant': np.ones(50),
            'skewed': np.random.exponential(1, 50),
        })

    def test_recommend_scaler_neural_network(self):
        """测试神经网络算法的标准化推荐"""
        selector = SmartScalerSelector(algorithm='neural_network')
        df = pd.DataFrame({
            'normal_col': np.random.normal(0, 1, 100),
            'constant_col': np.ones(100)
        })

        recommendations = selector.recommend_scaler(df)

        assert 'normal_col' in recommendations
        assert 'constant_col' in recommendations
        # 常数列应该被跳过
        assert recommendations['constant_col']['scaler'] in ['cons_skip']

    def test_recommend_scaler_tree_model(self):
        """测试树模型的标准化推荐"""
        selector = SmartScalerSelector(algorithm='random_forest')
        df = pd.DataFrame({
            'col1': np.random.normal(0, 1, 50)
        })

        recommendations = selector.recommend_scaler(df)
        assert 'col1' in recommendations


    def test_recommend_scaler_tree_model_noneed(self):
        """测试树模型的标准化推荐"""
        config_dict = {
            'transformers': [
                {'minmax': {'columns': ['col1']}},
                {'standard': {'columns': ['col2']}}
            ],
            'skip_scale': ['col3']
        }
        result = UnifiedFeatureScaler(method_config = config_dict,algorithm='decision_tree',pass_through=False)
        df = pd.DataFrame({
            'col1': np.random.normal(0, 1, 50)
        })

        df_transformed =  result.fit_transform(df,y=None)
        pd.testing.assert_frame_equal(df,df_transformed)



class TestStatisticsCalculation:
    """测试 StatisticsCalculation 统计计算"""

    def test_calculate_statistics_normal_data(self):
        """测试正常数据的统计计算"""
        calculator = StatisticsCalculation()
        df = pd.DataFrame({
            'normal': np.random.normal(0, 1, 100),
            'constant': np.ones(100)
        })

        stats = calculator.calculate_statistics_safely(df)

        assert 'normal' in stats
        assert 'constant' in stats
        assert stats['constant']['is_constant'] is True
        assert stats['normal']['is_constant'] is False

    def test_calculate_statistics_empty_data(self):
        """测试空数据统计计算"""
        calculator = StatisticsCalculation()
        df = pd.DataFrame({
            'empty': [np.nan, np.nan, np.nan]
        })

        stats = calculator.calculate_statistics_safely(df)
        assert stats['empty']['n_samples'] == 0