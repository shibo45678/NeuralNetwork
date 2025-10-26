import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
import sys
import os

# 添加模块路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.feature_engineer.split_datasets import TimeSeriesSplitter, CommonSplitter


class TestTimeSeriesSplitter:
    """时间序列分割器单元测试"""

    def test_initialization_with_valid_parameters(self):
        """测试使用有效参数初始化"""
        splitter = TimeSeriesSplitter(
            train_size=0.6,
            val_size=0.2,
            test_size=0.2,
            shuffle=False,
            random_state=42
        )

        assert splitter.train_size == 0.6
        assert splitter.val_size == 0.2
        assert splitter.test_size == 0.2
        assert splitter.shuffle == False
        assert splitter.random_state == 42

    def test_initialization_with_default_parameters(self):
        """测试使用默认参数初始化"""
        splitter = TimeSeriesSplitter()

        assert splitter.train_size == 0.7
        assert splitter.val_size == 0.2
        assert splitter.test_size == 0.1
        assert splitter.shuffle == False
        assert splitter.random_state is None

    def test_validation_rejects_invalid_total_size(self):
        """测试拒绝无效的比例总和"""
        with pytest.raises(ValueError, match="比例总和必须为1"):
            TimeSeriesSplitter(train_size=0.5, val_size=0.3, test_size=0.3)

    def test_split_numpy_array_without_y(self):
        """测试分割numpy数组（无标签）"""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        splitter = TimeSeriesSplitter(train_size=0.6, val_size=0.2, test_size=0.2)

        X_train, X_val, X_test = splitter.split(X)

        assert len(X_train) == 3  # 5 * 0.6 = 3
        assert len(X_val) == 1  # 5 * 0.2 = 1
        assert len(X_test) == 1  # 5 * 0.2 = 1

        # 验证时间序列顺序保持
        np.testing.assert_array_equal(X_train, np.array([[1, 2], [3, 4], [5, 6]]))
        np.testing.assert_array_equal(X_val, np.array([[7, 8]]))
        np.testing.assert_array_equal(X_test, np.array([[9, 10]]))

    def test_split_numpy_array_with_y(self):
        """测试分割numpy数组（有标签）"""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([10, 20, 30, 40, 50])
        splitter = TimeSeriesSplitter(train_size=0.6, val_size=0.2, test_size=0.2)

        X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X, y)

        assert len(X_train) == len(y_train) == 3
        assert len(X_val) == len(y_val) == 1
        assert len(X_test) == len(y_test) == 1

        # 验证数据标签对齐
        np.testing.assert_array_equal(X_train.flatten(), [1, 2, 3])
        np.testing.assert_array_equal(y_train, [10, 20, 30])

    def test_split_pandas_dataframe(self):
        """测试分割pandas DataFrame"""
        X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5], 'feature2': [10, 20, 30, 40, 50]})
        y = pd.Series([0, 1, 0, 1, 0])
        splitter = TimeSeriesSplitter(train_size=0.6, val_size=0.2, test_size=0.2)

        X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X, y)

        assert len(X_train) == 3
        assert len(X_val) == 1
        assert len(X_test) == 1

        # 验证索引重置
        assert list(X_train.index) == [0, 1, 2]
        assert list(y_train.index) == [0, 1, 2]

    def test_shuffle_disabled_for_time_series(self):
        """测试时间序列分割器禁用打乱功能"""
        X = np.array([[1], [2], [3], [4]])
        splitter = TimeSeriesSplitter(shuffle=True)  # 即使设置shuffle=True也不应该打乱

        X_train, X_val, X_test = splitter.split(X)

        # 时间序列应该保持原始顺序
        np.testing.assert_array_equal(X_train.flatten(), [1, 2])
        np.testing.assert_array_equal(X_val.flatten(), [3])

    def test_get_params_returns_correct_parameters(self):
        """测试get_params方法返回正确参数"""
        splitter = TimeSeriesSplitter(train_size=0.8, val_size=0.1, test_size=0.1, random_state=42)
        params = splitter.get_params()

        expected_params = {
            'train_size': 0.8,
            'val_size': 0.1,
            'test_size': 0.1,
            'shuffle': False,
            'random_state': 42
        }

        assert params == expected_params

    def test_set_params_updates_parameters(self):
        """测试set_params方法更新参数"""
        splitter = TimeSeriesSplitter()

        # 链式调用
        updated_splitter = splitter.set_params(
            train_size=0.5,
            val_size=0.3,
            test_size=0.2,
            shuffle=True,
            random_state=100
        )

        assert updated_splitter.train_size == 0.5
        assert updated_splitter.val_size == 0.3
        assert updated_splitter.test_size == 0.2
        assert updated_splitter.shuffle == True
        assert updated_splitter.random_state == 100
        assert updated_splitter is splitter  # 返回自身实例


class TestCommonSplitter:
    """通用分割器单元测试"""

    def test_shuffle_pandas_data(self):
        """测试打乱pandas数据"""
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        y = pd.Series([0, 1, 0, 1, 0])
        splitter = CommonSplitter(shuffle=True, random_state=42)

        X_shuffled, y_shuffled = splitter._shuffle_data(X, y)

        # 验证数据被打乱
        assert len(X_shuffled) == len(y_shuffled) == 5
        assert not X_shuffled.equals(X)  # 数据顺序改变
        assert not y_shuffled.equals(y)  # 标签顺序改变

        # 验证索引重置
        assert list(X_shuffled.index) == [0, 1, 2, 3, 4]
        assert list(y_shuffled.index) == [0, 1, 2, 3, 4]

        # 验证数据标签对齐
        for idx in range(len(X_shuffled)):
            original_idx = X.index[X['feature1'] == X_shuffled.iloc[idx]['feature1']][0]
            assert y_shuffled.iloc[idx] == y.iloc[original_idx]

    def test_shuffle_numpy_array(self):
        """测试打乱numpy数组"""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([10, 20, 30, 40, 50])
        splitter = CommonSplitter(shuffle=True, random_state=42)

        X_shuffled, y_shuffled = splitter._shuffle_data(X, y)

        assert len(X_shuffled) == len(y_shuffled) == 5
        # 验证数据被打乱但内容不变
        assert set(X_shuffled.flatten()) == {1, 2, 3, 4, 5}
        assert set(y_shuffled) == {10, 20, 30, 40, 50}

    def test_split_with_shuffle(self):
        """测试带打乱的分割"""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([10, 20, 30, 40, 50, 60])
        splitter = CommonSplitter(
            train_size=0.5,
            val_size=0.3,
            test_size=0.2,
            shuffle=True,
            random_state=42
        )

        X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X, y)

        # 验证分割比例
        assert len(X_train) == 3  # 6 * 0.5 = 3
        assert len(X_val) == 2  # 6 * 0.3 = 1.8 ≈ 2
        assert len(X_test) == 1  # 6 * 0.2 = 1.2 ≈ 1

        # 验证数据被打乱（与原始顺序不同）
        original_order = [1, 2, 3, 4, 5, 6]
        train_values = X_train.flatten()
        assert not all(train_values[i] == original_order[i] for i in range(len(train_values)))

    def test_split_pandas_with_shuffle(self):
        """测试带打乱的pandas数据分割"""
        X = pd.DataFrame({'feature': [1, 2, 3, 4, 5]})
        y = pd.Series([10, 20, 30, 40, 50])
        splitter = CommonSplitter(shuffle=True, random_state=42)

        X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X, y)

        # 验证索引重置
        assert list(X_train.index) == [0, 1, 2]
        assert list(y_train.index) == [0, 1, 2]
        assert list(X_val.index) == [0]
        assert list(y_val.index) == [0]
        assert list(X_test.index) == [0]
        assert list(y_test.index) == [0]


class TestIntegration:
    """集成测试"""

    def test_integration_with_sklearn_pipeline(self):
        """测试与sklearn Pipeline的集成"""
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        # 创建模拟数据
        X = np.random.randn(100, 3)
        y = np.random.randint(0, 2, 100)

        # 创建包含自定义分割器的pipeline
        pipeline = Pipeline([
            ('splitter', CommonSplitter(shuffle=True, random_state=42)),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42))
        ])

        # 测试pipeline参数访问
        params = pipeline.get_params()
        assert 'splitter' in params
        assert 'splitter__shuffle' in params

        # 测试参数设置
        pipeline.set_params(splitter__random_state=100)
        assert pipeline.named_steps['splitter'].random_state == 100

    def test_compatibility_with_grid_search(self):
        """测试与网格搜索的兼容性"""
        X = np.random.randn(50, 2)
        y = np.random.randint(0, 2, 50)

        splitter = CommonSplitter(shuffle=True, random_state=42)

        # 定义参数网格
        param_grid = {
            'train_size': [0.6, 0.7],
            'val_size': [0.2, 0.15],
            'shuffle': [True, False]
        }

        # 测试参数网格验证
        from sklearn.model_selection import ParameterGrid
        grid = ParameterGrid(param_grid)

        for params in grid:
            # 验证参数组合有效
            splitter.set_params(**params)
            # 测试分割不会报错
            result = splitter.split(X, y)
            assert len(result) == 6  # X_train, X_val, X_test, y_train, y_val, y_test

    def test_learn_process_interface(self):
        """测试learn和process接口"""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 1, 0, 1, 0])
        splitter = CommonSplitter()

        # 测试learn方法
        splitter_learned = splitter.learn(X, y)
        assert splitter_learned is splitter

        # 测试process方法
        result = splitter.process(X, y)
        assert len(result) == 6


class TestEdgeCases:
    """边界情况测试"""

    def test_single_sample(self):
        """测试单样本情况"""
        X = np.array([[1]])
        y = np.array([0])
        splitter = CommonSplitter()

        with pytest.raises(Exception):  # 应该处理单样本情况
            splitter.split(X, y)

    def test_empty_data(self):
        """测试空数据"""
        X = np.array([]).reshape(0, 2)
        y = np.array([])
        splitter = CommonSplitter()

        with pytest.raises(Exception):
            splitter.split(X, y)

    def test_extreme_split_ratios(self):
        """测试极端分割比例"""
        X = np.array([[1], [2], [3], [4], [5]])

        # 测试极小验证集
        splitter = CommonSplitter(train_size=0.98, val_size=0.01, test_size=0.01)
        result = splitter.split(X)
        assert len(result) == 3

        # 测试零测试集
        splitter = CommonSplitter(train_size=0.8, val_size=0.2, test_size=0.0)
        result = splitter.split(X)
        assert len(result[2]) == 0  # 测试集为空


class TestDataQuality:
    """数据质量测试"""

    def test_data_integrity_after_shuffle(self):
        """测试打乱后数据完整性"""
        X_original = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e']
        })
        y_original = pd.Series([10, 20, 30, 40, 50])

        splitter = CommonSplitter(shuffle=True, random_state=42)
        X_shuffled, y_shuffled = splitter._shuffle_data(X_original, y_original)

        # 验证数据内容不变
        assert set(X_original['A']) == set(X_shuffled['A'])
        assert set(X_original['B']) == set(X_shuffled['B'])
        assert set(y_original) == set(y_shuffled)

        # 验证数据类型不变
        assert X_shuffled.dtypes.equals(X_original.dtypes)

    def test_deterministic_shuffle_with_random_state(self):
        """测试随机种子保证结果确定性"""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([10, 20, 30, 40, 50])

        splitter1 = CommonSplitter(shuffle=True, random_state=42)
        splitter2 = CommonSplitter(shuffle=True, random_state=42)

        X1, y1 = splitter1._shuffle_data(X, y)
        X2, y2 = splitter2._shuffle_data(X, y)

        # 相同随机种子应该产生相同结果
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])