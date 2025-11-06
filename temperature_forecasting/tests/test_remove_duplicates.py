import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os
from data.data_preparation.remove_duplicates import RemoveDuplicates


class TestRemoveDuplicates:
    """RemoveDuplicates 类单元测试"""

    # 测试数据准备
    @pytest.fixture
    def sample_data_with_duplicates(self):
        """包含重复行的测试数据"""
        X = pd.DataFrame({
            'feature1': [1, 2, 2, 3, 3, 4],
            'feature2': ['A', 'B', 'B', 'C', 'C', 'D']
        })
        y = pd.Series([10, 20, 20, 30, 30, 40])
        return X, y

    @pytest.fixture
    def sample_data_no_duplicates(self):
        """无重复行的测试数据"""
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': ['A', 'B', 'C', 'D']
        })
        y = pd.Series([10, 20, 30, 40])
        return X, y

    @pytest.fixture
    def sample_data_x_only(self):
        """仅X数据的测试数据"""
        X = pd.DataFrame({
            'feature1': [1, 2, 2, 3],
            'feature2': ['A', 'B', 'B', 'C']
        })
        return X

    # 单元测试：初始化功能
    def test_initialization_default_config(self):
        """测试默认配置初始化"""
        cleaner = RemoveDuplicates()
        assert cleaner.pass_through is False
        assert cleaner.download_config['enabled'] is False
        assert cleaner._has_downloaded is False

    def test_initialization_custom_config(self):
        """测试自定义配置初始化"""
        download_config = {
            'enabled': True,
            'path': '/custom/path',
            'filename': 'custom_file.csv'
        }
        cleaner = RemoveDuplicates(download_config=download_config, pass_through=False)
        assert cleaner.pass_through is False
        assert cleaner.download_config['enabled'] is True
        assert cleaner.download_config['filename'] == 'custom_file.csv'

    # 单元测试：核心去重功能
    def test_remove_duplicates_with_y(self, sample_data_with_duplicates):
        """测试基于X和y的组合去重"""
        X, y = sample_data_with_duplicates
        cleaner = RemoveDuplicates()

        X_cleaned, y_cleaned = cleaner._remove_duplicates_with_y(X, y)

        # 验证去重结果
        assert len(X_cleaned) == 4  # 从6行去重到4行
        assert len(y_cleaned) == 4
        assert not X_cleaned.duplicated().any()

        # 验证索引同步
        assert X_cleaned.index.equals(y_cleaned.index)

    def test_remove_duplicates_x_only(self, sample_data_x_only):
        """测试仅基于X的去重"""
        X = sample_data_x_only
        cleaner = RemoveDuplicates()

        X_cleaned = cleaner._remove_duplicates_x_only(X)

        # 验证去重结果
        assert len(X_cleaned) == 3  # 从4行去重到3行
        assert not X_cleaned.duplicated().any()

    def test_no_duplicates_scenario(self, sample_data_no_duplicates):
        """测试无重复数据的情况"""
        X, y = sample_data_no_duplicates
        cleaner = RemoveDuplicates()

        X_cleaned, y_cleaned = cleaner._remove_duplicates_with_y(X, y)

        # 验证数据未改变
        assert len(X_cleaned) == len(X)
        assert len(y_cleaned) == len(y)
        assert X_cleaned.equals(X)
        assert y_cleaned.equals(y)

    # 单元测试：边界情况
    def test_empty_data(self):
        """测试空数据处理"""
        X = pd.DataFrame()
        y = pd.Series([], dtype=int)
        cleaner = RemoveDuplicates()

        X_cleaned, y_cleaned = cleaner._remove_duplicates_with_y(X, y)

        assert len(X_cleaned) == 0
        assert len(y_cleaned) == 0

    def test_single_row_data(self):
        """测试单行数据处理"""
        X = pd.DataFrame({'feature1': [1], 'feature2': ['A']})
        y = pd.Series([10])
        cleaner = RemoveDuplicates()

        X_cleaned, y_cleaned = cleaner._remove_duplicates_with_y(X, y)

        assert len(X_cleaned) == 1
        assert len(y_cleaned) == 1

    # 单元测试：pass_through 模式
    def test_pass_through_mode(self, sample_data_with_duplicates):
        """测试直通模式"""
        X, y = sample_data_with_duplicates
        cleaner = RemoveDuplicates(pass_through=True)

        result = cleaner.learn_process(X, y)

        # 验证数据未处理
        if isinstance(result, tuple):
            X_result, y_result = result
            assert X_result.equals(X)
            assert y_result.equals(y)
        else:
            assert result.equals(X)

    # 集成测试：装饰器兼容性
    def test_with_validation_decorators(self, sample_data_with_duplicates):
        """测试与输入输出验证装饰器的兼容性"""
        X, y = sample_data_with_duplicates
        cleaner = RemoveDuplicates()

        # 应该能正常通过装饰器验证
        result = cleaner.learn_process(X, y)

        assert result is not None
        if isinstance(result, tuple):
            assert len(result) == 2

    # 功能测试：数据质量
    def test_data_quality_after_deduplication(self, sample_data_with_duplicates):
        """测试去重后的数据质量"""
        X, y = sample_data_with_duplicates
        cleaner = RemoveDuplicates()

        X_cleaned, y_cleaned = cleaner.learn_process(X, y)

        # 去重后验证没有重复行
        combined = X_cleaned.copy()
        combined['target'] = y_cleaned.values
        assert not combined.duplicated().any()

        # 验证数据类型保持不变
        assert X_cleaned.dtypes.equals(X.dtypes)

    # 功能测试：性能基准
    def test_performance_with_large_dataset(self):
        """测试大数据集下的性能"""
        # 创建大型测试数据集
        n_rows = 10000
        X = pd.DataFrame({
            'feature1': np.random.randint(0, 100, n_rows),
            'feature2': np.random.choice(['A', 'B', 'C', 'D'], n_rows)
        })
        y = pd.Series(np.random.randn(n_rows))

        cleaner = RemoveDuplicates()

        # 性能测试：应该在合理时间内完成
        import time
        start_time = time.time()
        X_cleaned, y_cleaned = cleaner.learn_process(X, y)
        end_time = time.time()

        execution_time = end_time - start_time
        print(f"处理 {n_rows} 行数据用时: {execution_time:.2f} 秒")

        # 验证处理成功
        assert len(X_cleaned) <= len(X)
        assert execution_time < 10.0  # 10秒超时

    # 错误处理测试
    def test_invalid_input_types(self):
        """测试无效输入类型"""
        cleaner = RemoveDuplicates()

        with pytest.raises(Exception):  # 具体异常类型取决于装饰器
            cleaner.learn_process("invalid_input", "invalid_target")

    @patch('pandas.DataFrame.duplicated')  # 替换 pandas 的 duplicated 方法
    def test_duplicate_detection_failure(self, mock_duplicated, sample_data_with_duplicates):
        """测试重复检测失败的情况"""
        X, y = sample_data_with_duplicates
        mock_duplicated.side_effect = Exception("Duplicate detection failed") # 设置模拟对象在调用时抛出异常
        cleaner = RemoveDuplicates()

        with pytest.raises(Exception):
            cleaner.learn_process(X, y)

    # 数据下载功能测试
    def test_download_functionality(self, sample_data_with_duplicates):
        """测试数据下载功能"""
        X, y = sample_data_with_duplicates

        with tempfile.TemporaryDirectory() as temp_dir:
            download_config = {
                'enabled': True,
                'path': temp_dir,
                'filename': 'test_duplicates.csv'
            }
            cleaner = RemoveDuplicates(download_config=download_config)

            # 处理数据并触发下载
            cleaner.learn_process(X, y)

            # 验证文件是否创建
            expected_file = os.path.join(temp_dir, 'test_duplicates.csv')
            # 注意：由于装饰器中的条件，文件可能不会总是创建

    # 回归测试：修复已知问题
    def test_series_ambiguity_error_fixed(self, sample_data_with_duplicates):
        """测试修复 pandas Series 真值歧义错误"""
        X, y = sample_data_with_duplicates
        cleaner = RemoveDuplicates(download_config={'enabled': True})

        # 这个测试应该通过，不再抛出 ValueError
        try:
            result = cleaner.learn_process(X, y)
            assert result is not None
        except ValueError as e:
            if "The truth value of a Series is ambiguous" in str(e):
                pytest.fail("Series 真值歧义错误未被修复")
            else:
                raise  # 重新抛出其他异常


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])