import pytest
import pandas as pd
import numpy as np
from data.data_cleaner.sampling import SystematicResampler,TimeBasedResampler

class TestSystematicResampler:
    """SystematicResampler 单元测试"""

    def test_systematic_resample_basic(self):
        """测试基本系统抽样功能"""
        # 准备测试数据
        X = pd.DataFrame({
            'feature1': range(20),
            'feature2': range(20, 40)
        })
        y = pd.Series(range(20))

        # 创建采样器
        sampler = SystematicResampler(start_index=2, step=3)

        # 测试 learn_process
        X_sampled, y_sampled = sampler.learn_process(X, y)

        # 验证结果
        expected_indices = [2, 5, 8, 11, 14, 17]
        assert len(X_sampled) == len(expected_indices)
        assert len(y_sampled) == len(expected_indices)
        assert list(X_sampled.index) == expected_indices
        assert list(y_sampled.index) == expected_indices

    def test_systematic_resample_without_y(self):
        """测试无目标变量的系统抽样"""
        X = pd.DataFrame({'feature1': range(10)})

        sampler = SystematicResampler(start_index=1, step=2)
        X_sampled = sampler.learn_process(X)

        expected_indices = [1, 3, 5, 7, 9]
        assert len(X_sampled) == len(expected_indices)
        assert list(X_sampled.index) == expected_indices

    def test_systematic_resample_mismatched_indices(self):
        """测试索引不一致但长度一致的情况"""
        # X 和 y 有相同长度但不同索引
        X = pd.DataFrame({'feature1': range(10)}, index=range(10, 20))
        y = pd.Series(range(10), index=range(20, 30))  # 不同索引

        sampler = SystematicResampler(start_index=0, step=2)

        # 应该能正确处理，因为采样基于位置而非索引
        X_sampled, y_sampled = sampler.learn_process(X, y)

        # 验证采样后的数据关系
        assert len(X_sampled) == len(y_sampled)
        # 验证数据值是否正确
        expected_X_values = [0, 2, 4, 6, 8]  # 因为start_index=0, step=2，所以选取的是这些位置上的值
        expected_y_values = [0, 2, 4, 6, 8]
        assert list(X_sampled['feature1']) == expected_X_values
        assert list(y_sampled) == expected_y_values
        # 注意：索引不同，所以不检查索引相等

    def test_systematic_resample_different_lengths(self):
        """测试 X 和 y 长度不一致的情况"""
        X = pd.DataFrame({'feature1': range(10)})
        y = pd.Series(range(8))  # 长度不一致

        sampler = SystematicResampler(start_index=0, step=2)

        with pytest.raises(ValueError, match="长度不一致"):
            sampler.learn_process(X, y)

    def test_systematic_resample_edge_cases(self):
        """测试边界情况"""
        # 空数据
        X_empty = pd.DataFrame({'feature1': []})
        y_empty = pd.Series([], dtype=int)

        sampler = SystematicResampler(start_index=0, step=1)
        with pytest.raises(ValueError, match="输入数据X不能全部为NaN"):
            X_result, y_result = sampler.learn_process(X_empty, y_empty)


        # 单样本数据
        X_single = pd.DataFrame({'feature1': [1]})
        y_single = pd.Series([0])

        X_result, y_result = sampler.learn_process(X_single, y_single)
        assert len(X_result) == 1
        assert len(y_result) == 1

    def test_systematic_resample_learn_process_separate(self):
        """测试分开调用 learn 和 process"""
        X = pd.DataFrame({'feature1': range(10)})
        y = pd.Series(range(10))

        sampler = SystematicResampler(start_index=1, step=2)

        # 先 learn
        sampler.learn(X, y)

        # 再 process
        X_sampled, y_sampled = sampler.process(X, y)

        expected_indices = [1, 3, 5, 7, 9]
        assert len(X_sampled) == len(expected_indices)
        assert len(y_sampled) == len(expected_indices)

    def test_systematic_resample_process_before_learn(self):
        """测试在 learn 之前调用 process 的异常"""
        X = pd.DataFrame({'feature1': range(10)})

        sampler = SystematicResampler()

        with pytest.raises(ValueError, match="请先调用 fit 方法"):
            sampler.process(X)


class TestTimeBasedResampler:
    """TimeBasedResampler 单元测试"""

    @pytest.fixture
    def sample_time_data(self):
        """创建样本时间序列数据"""
        dates = pd.date_range('2023-01-01', periods=24, freq='H')
        X = pd.DataFrame({
            'timestamp': dates,
            'value1': np.random.randn(24),
            'value2': np.random.randn(24)
        })
        y = pd.Series(np.random.randint(0, 2, 24), name='target')
        return X, y

    def test_time_based_resample_basic(self, sample_time_data):
        """测试基本时间重采样功能"""
        X, y = sample_time_data

        sampler = TimeBasedResampler(
            time_column='timestamp',
            freq='3H',
            aggregation='mean'
        )

        X_sampled, y_sampled = sampler.learn_process(X, y)

        # 24小时数据，3小时频率，应该有8个样本
        assert len(X_sampled) == 8
        assert len(y_sampled) == 8
        assert 'timestamp' in X_sampled.columns

    def test_time_based_resample_different_aggregations(self, sample_time_data):
        """测试不同的聚合方法"""
        X, y = sample_time_data

        # 测试 X 和 y 使用不同的聚合方法
        sampler = TimeBasedResampler(
            time_column='timestamp',
            freq='6H',
            aggregation='mean',
            y_aggregation='sum'  # y 使用求和
        )

        X_sampled, y_sampled = sampler.learn_process(X, y)

        assert len(X_sampled) == 4  # 24小时/6小时 = 4个区间
        # 验证聚合方法应用正确
        assert 'value1' in X_sampled.columns
        assert 'value2' in X_sampled.columns

    def test_time_based_resample_without_time_column(self):
        """测试没有时间列的情况"""
        X = pd.DataFrame({'feature1': range(10)})
        y = pd.Series(range(10))

        sampler = TimeBasedResampler(time_column='nonexistent')

        # 应该返回原始数据
        X_result, y_result = sampler.learn_process(X, y)

        pd.testing.assert_frame_equal(X, X_result)
        pd.testing.assert_series_equal(y, y_result)

    def test_time_based_resample_mismatched_indices(self):
        """测试索引范围 不一致的时间序列数据"""
        dates_x = pd.date_range('2023-01-01', periods=10, freq='H')

        X = pd.DataFrame({
            'timestamp': dates_x,
            'value': range(10)
        }, index=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18])  # 偶数索引

        y = pd.Series(range(10), index=[1, 3, 7, 9, 5, 11, 13, 17, 15, 19])  # 奇数索引

        sampler = TimeBasedResampler(
            time_column='timestamp',
            freq='2H',
            aggregation='mean'
        )

        # 应该能正常处理，pd.concat 会处理索引对齐
        X_sampled, y_sampled = sampler.learn_process(X, y)

        # 验证结果合理性
        assert len(X_sampled) > 0
        assert len(y_sampled) > 0
        assert len(X_sampled) == len(y_sampled)

    def test_time_based_resample_mismatched_indices2(self):
        """测试索引范围一致，顺序不一致的时间序列数据"""
        dates_x = pd.date_range('2023-01-01', periods=10, freq='H')

        X = pd.DataFrame({
            'timestamp': dates_x,
            'value': range(10)
        }, index=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18])  # 偶数索引

        y = pd.Series(range(10), index=[18, 14, 8, 6, 0, 2, 4, 16, 12, 10])  # 索引范围一致，但是索引顺序不一致

        sampler = TimeBasedResampler(
            time_column='timestamp',
            freq='2H',
            aggregation='mean'
        )

        # 应该能正常处理，pd.concat 会处理索引对齐
        X_sampled, y_sampled = sampler.learn_process(X, y)

        # 验证结果合理性
        assert len(X_sampled) > 0
        assert len(y_sampled) > 0
        assert len(X_sampled) == len(y_sampled)

    def test_time_based_resample_unsorted_data(self):
        """测试未排序的时间序列数据"""
        # 创建乱序的时间数据
        dates = pd.date_range('2023-01-01', periods=6, freq='H')
        shuffled_dates = dates.take([2, 0, 4, 1, 5, 3])  # 打乱顺序

        X = pd.DataFrame({
            'timestamp': shuffled_dates,
            'value': [2, 0, 4, 1, 5, 3]  # 值对应原始顺序
        })
        y = pd.Series([20, 0, 40, 10, 50, 30], name='target')

        sampler = TimeBasedResampler(
            time_column='timestamp',
            freq='2H',
            aggregation='mean'
        )

        X_sampled, y_sampled = sampler.learn_process(X, y)

        # 即使输入是乱序的，输出也应该是按时间排序的
        assert X_sampled['timestamp'].is_monotonic_increasing
        # 2小时间隔，应该有3个样本
        assert len(X_sampled) == 3

    def test_time_based_resample_edge_cases(self):
        """测试时间重采样的边界情况"""
        # 单时间点数据
        X_single = pd.DataFrame({
            'timestamp': [pd.Timestamp('2023-01-01')],
            'value': [1]
        })
        y_single = pd.Series([0])

        sampler = TimeBasedResampler(time_column='timestamp', freq='H')
        X_result, y_result = sampler.learn_process(X_single, y_single)

        assert len(X_result) == 1
        assert len(y_result) == 1

        # 空数据
        X_empty = pd.DataFrame({'timestamp': [], 'value': []})
        y_empty = pd.Series([], dtype=int)
        with pytest.raises(ValueError,match = "输入数据X不能全部为NaN"):
            X_result, y_result = sampler.learn_process(X_empty, y_empty)

    def test_time_based_resample_learn_process_separate(self, sample_time_data):
        """测试分开调用 learn 和 process"""
        X, y = sample_time_data

        sampler = TimeBasedResampler(
            time_column='timestamp',
            freq='4H',
            aggregation='mean'
        )

        # 先 learn
        sampler.learn(X, y)

        # 再 process
        X_sampled, y_sampled = sampler.process(X, y)

        assert len(X_sampled) == 6  # 24小时/4小时 = 6个区间
        assert len(y_sampled) == 6