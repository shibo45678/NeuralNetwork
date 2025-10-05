import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from data.processing import RemoveDuplicates




def test_remove_duplicates_basic():
    """基础功能测试"""
    print("=" * 60)
    print("基础功能测试")
    print("=" * 60)

    # 创建测试数据
    data = {
        'id': [1, 2, 3, 4, 5, 6, 7],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David', 'Eve'],
        'age': [25, 30, 30, 35, 40, 40, 45],
        'score': [85, 90, 90, 78, 92, 92, 88]
    }
    df = pd.DataFrame(data)

    print("原始数据:")
    print(df)
    print(f"原始数据形状: {df.shape}")

    # 测试1: 不启用下载
    print("\n1. 测试不启用下载功能:")
    remover1 = RemoveDuplicates()
    result1 = remover1.fit_transform(df)
    print(f"去重后形状: {result1.shape}")
    print("去重结果:")
    print(result1)

    # 测试2: 启用下载
    print("\n2. 测试启用下载功能:")
    with tempfile.TemporaryDirectory() as temp_dir:
        remover2 = RemoveDuplicates({
            'enabled': True,
            'path': temp_dir,
            'filename': 'test_duplicates.csv'
        })
        result2 = remover2.fit_transform(df)
        print(f"去重后形状: {result2.shape}")

        # 检查文件
        expected_file = os.path.join(temp_dir, 'test_duplicates.csv')
        if os.path.exists(expected_file):
            downloaded = pd.read_csv(expected_file)
            print(f"✓ 成功下载重复数据，包含 {len(downloaded)} 行")
        else:
            print("✗ 文件未创建")


def test_edge_cases():
    """边界情况测试"""
    print("\n" + "=" * 60)
    print("边界情况测试")
    print("=" * 60)

    # 测试无重复数据
    print("1. 无重复数据测试:")
    no_dup_data = {
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie']
    }
    no_dup_df = pd.DataFrame(no_dup_data)
    remover = RemoveDuplicates({'enabled': True})
    result = remover.fit_transform(no_dup_df)
    print(f"无重复数据处理后形状: {result.shape}")

    # 测试全重复数据
    print("\n2. 全重复数据测试:")
    all_dup_data = {
        'id': [1, 1, 1],
        'name': ['Alice', 'Alice', 'Alice']
    }
    all_dup_df = pd.DataFrame(all_dup_data)
    remover = RemoveDuplicates({'enabled': True})
    result = remover.fit_transform(all_dup_df)
    print(f"全重复数据处理后形状: {result.shape}")
    print("处理后数据:")
    print(result)


def test_pipeline_integration():
    """Pipeline集成测试"""
    print("\n" + "=" * 60)
    print("Pipeline集成测试")
    print("=" * 60)

    # 创建训练数据
    train_data = {
        'feature1': [1, 2, 2, 3, 4, 4, 5],
        'feature2': [1.1, 2.2, 2.2, 3.3, 4.4, 4.4, 5.5],
        'target': [0, 1, 1, 0, 1, 1, 0]
    }
    df = pd.DataFrame(train_data)
    X = df[['feature1', 'feature2']]
    y = df['target']
    print(f"X的shape{X.shape}")
    print(f"y的shape{y.shape}")
    # 创建pipeline
    pipeline = Pipeline([
        ('remove_duplicates', RemoveDuplicates()),
        ('classifier', RandomForestClassifier(n_estimators=5, random_state=42))
    ])


    # 训练
    pipeline.fit(X, y)
    print("✓ Pipeline训练成功")

    # 预测
    X_test = pd.DataFrame({
        'feature1': [2, 3, 4],
        'feature2': [2.2, 3.3, 4.4]
    })
    predictions = pipeline.predict(X_test)
    print(f"预测结果: {predictions}")


def test_error_handling():
    """错误处理测试"""
    print("\n" + "=" * 60)
    print("错误处理测试")
    print("=" * 60)

    # 测试空数据
    print("1. 空数据测试:")
    try:
        remover = RemoveDuplicates()
        remover.fit(None)
    except ValueError as e:
        print(f"✓ 正确捕获错误: {e}")

    # 测试空DataFrame
    print("\n2. 空DataFrame测试:")
    try:
        remover = RemoveDuplicates()
        empty_df = pd.DataFrame()
        remover.fit(empty_df)
    except ValueError as e:
        print(f"✓ 正确捕获错误: {e}")


if __name__ == "__main__":
    print("开始测试 RemoveDuplicates 转换器")

    test_remove_duplicates_basic()
    test_edge_cases()
    test_pipeline_integration()
    test_error_handling()

    print("\n" + "=" * 60)
    print("所有测试完成! ✓")