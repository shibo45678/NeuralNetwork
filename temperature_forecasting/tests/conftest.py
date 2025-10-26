# /Users/shibo/Python/NeuralNetwork/temperature_forecasting/tests/conftest.py
import sys
import os
import pytest
import pandas as pd
import numpy as np

# 在模块级别设置路径（确保在导入测试文件前执行）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')

if src_path not in sys.path:
    sys.path.insert(0, src_path)
    print(f"✅ conftest.py 已设置路径: {src_path}")


def pytest_configure(config):
    """pytest 配置钩子 - 可选"""
    print("✅ pytest 配置完成")

@pytest.fixture(scope="session")
def global_sample_data():
    """全局样本数据"""
    np.random.seed(42)
    data= pd.DataFrame({
        'normal_col': np.random.normal(0, 1, 100),
        'constant_col': np.ones(100,dtype=np.float64),
        'skewed_col': np.random.exponential(2, 100).astype(np.float64),
        'outlier_col': np.concatenate(
            [np.random.normal(0, 1, 95),
             np.random.normal(10, 1, 5)]).astype(np.float64),
        'uniform_col': np.random.uniform(0, 100, 100).astype(np.float64)
    })
    return data
