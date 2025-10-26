import pytest
import pandas as pd
import numpy as np
from pydantic import ValidationError

from data.data_cleaner.sampling import SystematicResampler,TimeBasedResampler



class TestResamplerFunctional:
    """采样器功能测试"""

    def test_performance_with_large_data(self):
        """测试大数据量下的性能"""
        # 创建大数据集
        n_samples = 10000
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples)
        })
        y = pd.Series(np.random.randint(0, 2, n_samples))

        sampler = SystematicResampler(start_index=0, step=10)

        import time
        start_time = time.time()
        X_sampled, y_sampled = sampler.learn_process(X, y)
        end_time = time.time()

        # 性能检查：应该在合理时间内完成
        processing_time = end_time - start_time
        assert processing_time < 5.0  # 5秒内完成10000个样本的抽样

        # 验证采样正确性
        assert len(X_sampled) == n_samples // 10
        assert len(y_sampled) == n_samples // 10

    def test_data_quality_preservation(self):
        """测试数据质量保持"""
        # 创建有特殊值的数据
        X = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        sampler = SystematicResampler(start_index=0, step=2)
        X_sampled, y_sampled = sampler.learn_process(X, y)

        # 验证采样后数据质量
        assert not X_sampled.empty
        assert X_sampled.isna().sum().sum() <= X.isna().sum().sum()  # NaN 数量不应增加

    def test_business_scenarios(self):
        """测试业务场景"""
        # 场景1：时间序列预测
        dates = pd.date_range('2023-01-01', periods=168, freq='H')  # 一周的数据
        X_business = pd.DataFrame({
            'timestamp': dates,
            'sales': np.random.poisson(100, 168),  # 模拟销售数据
            'promotion': np.random.randint(0, 2, 168)  # 促销活动
        })
        y_business = pd.Series(np.random.poisson(50, 168), name='returns')  # 退货数量

        # 按天聚合
        sampler = TimeBasedResampler(
            time_column='timestamp',
            freq='D',
            aggregation='sum'  # 每日总和
        )
        X_daily, y_daily = sampler.learn_process(X_business, y_business)

        # 验证业务逻辑
        assert len(X_daily) == 7  # 7天
        assert 'sales' in X_daily.columns
        assert 'promotion' in X_daily.columns
        assert X_daily['sales'].sum() > 0  # 销售总额应为正

    def test_error_handling_and_validation(self):
        """测试错误处理和验证"""
        # 测试无效参数
        with pytest.raises(ValidationError):
            s = SystematicResampler(start_index="invalid")  # 应该需要整数

        # 测试空数据
        X_empty = pd.DataFrame()
        y_empty = pd.Series([], dtype=int)

        sampler = SystematicResampler()
        # 应该能处理空数据而不崩溃
        with pytest.raises(ValueError,match="输入数据X不能全部为NaN"):
         X_result, y_result = sampler.learn_process(X_empty, y_empty)


        # 测试无效的时间频率
        X_time = pd.DataFrame({
            'timestamp': [pd.Timestamp('2023-01-01')],
            'value': [1]
        })

        invalid_sampler = TimeBasedResampler(
            time_column='timestamp',
            freq='INVALID_FREQ'
        )

        # 应该能优雅地处理无效频率
        try:
            invalid_sampler.learn_process(X_time)
        except Exception as e:
            # 应该给出有意义的错误信息
            assert '频率' in str(e) or 'freq' in str(e).lower()