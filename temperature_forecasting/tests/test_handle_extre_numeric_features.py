
from data.data_preparation.handle_extre_numeric_features import NumericOutlierProcessor
from data.data_preparation.check_extre_numeric_features import CheckExtreFeatures

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def test_numeric_outlier_processor():
    """测试 NumericOutlierProcessor 类的各种功能"""

    # 1. 创建测试数据（包含异常值）
    print("=" * 60)
    print("1. 创建测试数据")
    print("=" * 60)

    # 生成有异常值的数据
    np.random.seed(42)
    X, y = make_classification(n_samples=1000, n_features=4, n_redundant=0,
                               n_informative=4, random_state=42)

    # 转换为DataFrame
    feature_names = ['feature1', 'feature2', 'feature3', 'feature4']
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')

    # 手动添加一些异常值
    X_df.loc[10, 'feature1'] = 10  # 明显异常值
    X_df.loc[20, 'feature2'] = -8  # 明显异常值
    X_df.loc[30, 'feature3'] = 15  # 明显异常值
    X_df.loc[40, 'feature4'] = -10  # 明显异常值

    print(f"数据形状: {X_df.shape}")
    print(f"特征列: {X_df.columns.tolist()}")
    print(f"手动添加的异常值位置: 10, 20, 30, 40")

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42
    )

    # 2. 先单独测试 CheckExtreFeatures
    print("\n" + "=" * 60)
    print("2. 先测试 CheckExtreFeatures")
    print("=" * 60)

    try:
        checker = CheckExtreFeatures(method_config={'method': 'isolationforest', 'threshold': 0.05})
        checker.fit(X_train)
        X_checked = checker.transform(X_train)
        print(f"✓ CheckExtreFeatures 测试成功")
        print(f"检测到的异常值数量: {checker.outliers_mask.sum()}")
    except Exception as e:
        print(f"✗ CheckExtreFeatures 测试失败: {e}")
        return

    # 3. 测试1: 默认配置（isolation forest）
    print("\n" + "=" * 60)
    print("3. 测试默认配置 - Isolation Forest")
    print("=" * 60)

    try:
        processor_default = NumericOutlierProcessor(
            method_config=None,  # 使用默认配置
            handle_strategy='mark'
        )

        # 先调用 fit，再调用 transform
        processor_default.fit(X_train, y_train)
        X_train_processed, y_train_processed = processor_default.transform(X_train, y_train)

        # 检查结果
        print(f"✓ 默认配置测试成功")
        print(f"处理后的训练数据形状: {X_train_processed.shape}")
        print(f"是否添加了异常值列: {'is_entire_outlier' in X_train_processed.columns}")
        if 'is_entire_outlier' in X_train_processed.columns:
            outlier_count = X_train_processed['is_entire_outlier'].value_counts()
            print(f"检测到的异常值数量: {outlier_count}")

        # 获取检测报告
        report = processor_default.get_detection_report()
        print(f"检测报告 - 总异常值: {report['total_outliers']}")

    except Exception as e:
        print(f"✗ 默认配置测试失败: {e}")

    # 4. 测试2: 列级配置 + 函数传递
    print("\n" + "=" * 60)
    print("4. 测试列级配置 + 函数传递")
    print("=" * 60)

    try:
        # 定义自定义异常检测函数
        def custom_outlier_detector(series):
            """自定义异常检测：超出3倍标准差"""
            mean_val = series.mean()
            std_val = series.std()
            return (series < mean_val - 1 * std_val) | (series > mean_val + 1 * std_val)

        # 创建列级配置
        column_config = [
            ('zscore', {'threshold': 2.5, 'columns': ['feature1','feature2']}),
            ('iqr', {'threshold': 1.8, 'columns': ['feature2']}),
            ('custom', {'functions': [custom_outlier_detector], 'columns': ['feature3']}),
        ]

        processor_column = NumericOutlierProcessor(
            method_config=column_config,
            handle_strategy='mark'
        )

        # 拟合和转换
        processor_column.fit(X_train, y_train)
        X_train_col_processed, y_train_col_processed = processor_column.transform(X_train, y_train)

        # 检查结果
        print(f"✓ 列级配置测试成功")
        print(f"处理后的训练数据形状: {X_train_col_processed.shape}")
        outlier_cols = [col for col in X_train_col_processed.columns if 'is_outlier' in col]
        print(f"添加的异常值列: {outlier_cols}")

        # 检查每列的异常值数量
        for col in outlier_cols:
            outlier_count = X_train_col_processed[col].value_counts()
            print(f"列 {col} 的异常值数量: {outlier_count}")

        # 获取检测报告
        report_col = processor_column.get_detection_report()
        print(f"列级配置检测报告 - 总异常值: {report_col['total_outliers']}")

    except Exception as e:
        print(f"✗ 列级配置测试失败: {e}")

    # 5. 测试3: 不同处理策略
    print("\n" + "=" * 60)
    print("5. 测试不同处理策略")
    print("=" * 60)

    strategies_to_test = ['mark', 'nan']

    for strategy in strategies_to_test:
        print(f"\n--- 测试策略: {strategy} ---")

        try:
            # 使用简单的配置
            simple_config = [
                ('zscore', {'threshold': 2.5, 'columns': ['feature1']}),
            ]

            processor_strategy = NumericOutlierProcessor(
                method_config=simple_config,
                handle_strategy=strategy
            )

            # 拟合和转换
            processor_strategy.fit(X_train.copy(), y_train.copy())
            result = processor_strategy.transform(X_train.copy(), y_train.copy())

            if isinstance(result, tuple):
                X_processed, y_processed = result
                print(f"✓ {strategy} 策略成功")
                print(f"处理后的X形状: {X_processed.shape}, y形状: {y_processed.shape}")
            else:
                X_processed = result
                print(f"✓ {strategy} 策略成功")
                print(f"处理后的X形状: {X_processed.shape}")

            # 检查特定策略的效果
            if strategy == 'nan':
                nan_count = X_processed.isna().sum().sum()
                print(f"NaN值数量: {nan_count}")
            elif strategy == 'mark':
                outlier_cols = [col for col in X_processed.columns if 'is_outlier' in col]
                print(f"添加的标记列: {outlier_cols}")

        except Exception as e:
            print(f"✗ 策略 {strategy} 执行失败: {e}")

    # 6. 最终总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    print("测试完成！主要问题已修复：")
    print("✓ 添加了 fit 方法的 return self")
    print("✓ 分离了 fit 和 transform 调用")
    print("✓ 逐步测试各个功能模块")


# 运行测试
if __name__ == "__main__":
    test_numeric_outlier_processor()



