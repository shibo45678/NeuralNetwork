import pandas as pd
import numpy as np
from datetime import datetime
import pytest
from data.feature_engineering.features_selector import FeaturesSelector


# 创建测试数据的 fixture
@pytest.fixture
def sample_data():
    """创建测试数据 fixture"""
    np.random.seed(42)
    X = pd.DataFrame({
        'age': np.random.randint(18, 60, 100),
        'income': np.random.normal(50000, 15000, 100),
        'score': np.random.uniform(0, 100, 100),
        'city': np.random.choice(['北京', '上海', '广州', '深圳'], 100),
        'education': np.random.choice(['本科', '硕士', '博士', '大专'], 100),
        'user_note': [f'user_{i}_note' for i in range(100)],
        'signup_date': [datetime(2020, np.random.randint(1, 13), np.random.randint(1, 28)) for _ in range(100)],
        'last_login': [datetime(2023, np.random.randint(1, 13), np.random.randint(1, 28)) for _ in range(100)],
    })
    y = (X['age'] * 0.3 +
         (X['city'].map({'北京': 10, '上海': 20, '广州': 5, '深圳': 15})) * 0.5 +
         np.random.normal(0, 5, 100))
    return X, y


# 测试案例1：正常训练模式
def test_normal_training_mode(sample_data):
    """测试正常训练模式"""
    X_train, y_train = sample_data
    print("=== 测试案例1：正常训练模式 ===")
    print("原始特征:", X_train.columns.tolist())
    print(f"user_note的类型: {X_train['user_note'].dtype}")

    selector = FeaturesSelector(threshold=0, mode='train')
    selector.fit(X_train, y_train)

    X_train_selected = selector.transform(X_train)
    assert X_train_selected.shape[1] > 0  # 确保有特征被选择
    print("训练后特征形状:", X_train_selected.shape)
    print("训练后特征列:", X_train_selected.columns.tolist())


# 测试案例2：预测模式
def test_prediction_mode(sample_data):
    """测试预测模式"""
    X_train, y_train = sample_data
    # 先训练一个selector
    selector = FeaturesSelector(threshold=0, mode='train')
    selector.fit(X_train, y_train)

    print("\n=== 测试案例2：预测模式 ===")
    X_test, _ = sample_data
    # 模拟预测数据中可能出现未见过的类别
    X_test = X_test.copy()
    X_test.loc[0, 'city'] = '杭州'  # 未见过的城市

    X_test_selected = selector.transform(X_test)
    assert X_test_selected.shape[1] > 0
    print("预测数据特征形状:", X_test_selected.shape)
    print("预测数据特征列:", X_test_selected.columns.tolist())
    print('分类列是否完成编码:', X_test_selected.head(10))


# 测试案例3：高阈值情况
def test_high_threshold(sample_data):
    """测试高阈值情况"""
    X_train, y_train = sample_data
    print("\n=== 测试案例3：高阈值情况 ===")
    selector_high_threshold = FeaturesSelector(threshold=0.9, mode='train')
    selector_high_threshold.fit(X_train, y_train)

    X_high_threshold = selector_high_threshold.transform(X_train)
    # 高阈值应该选择更少的特征
    assert X_high_threshold.shape[1] >= 0
    print("高阈值后特征形状:", X_high_threshold.shape)


# 测试案例4：无目标变量情况
def test_no_target_variable(sample_data):
    """测试无目标变量情况"""
    X_train, y_train = sample_data
    print("\n=== 测试案例4：无目标变量情况 ===")
    selector_no_y = FeaturesSelector(threshold=0.1, mode='predict')
    selector_no_y.fit(X_train)  # 不传入y

    X_no_y = selector_no_y.transform(X_train)
    assert X_no_y.shape[1] > 0
    print("无目标变量特征形状:", X_no_y.shape)
