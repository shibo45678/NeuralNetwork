import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
from data.feature_engineer.scalers import UnifiedFeatureScaler, SmartScalerSelector, AlgorithmAwareScalerSelector

warnings.filterwarnings('ignore')


def test_algorithm_aware_selector():
    """测试 AlgorithmAwareScalerSelector 的功能"""
    print("=== 测试 AlgorithmAwareScalerSelector ===")

    selector = AlgorithmAwareScalerSelector()

    # 测试数据特征
    test_stats = [
        {
            'name': '正态分布数据',
            'stats': {'scaler': 'standard', 'outlier_ratio_iqr': 0.01, 'skewness': 0.2,'n_samples':100},
            'algorithms': ['neural_network', 'svm', 'random_forest']
        },
        {
            'name': '高异常值数据',
            'stats': {'scaler': 'standard', 'outlier_ratio_iqr': 0.15, 'skewness': 0.5,'n_samples':100},
            'algorithms': ['neural_network', 'svm', 'linear_regression']
        },
        {
            'name': '高偏度数据',
            'stats': {'scaler': 'standard', 'outlier_ratio_iqr': 0.02, 'skewness': 3.5,'n_samples':100},
            'algorithms': ['neural_network', 'knn', 'xgboost']
        },
        {
            'name': '跳过数据',
            'stats': {'scaler': 'skip','n_samples':1},
            'algorithms': ['neural_network', 'svm']
        }
    ]

    for test_case in test_stats:
        print(f"\n--- {test_case['name']} ---")
        for algorithm in test_case['algorithms']:
            scaler_type, reason = selector.recommend_for_algorithm(algorithm, test_case['stats'])
            if scaler_type :
                print(f"  算法 {algorithm:20} -> 推荐: {scaler_type:10} | 理由: {reason}")

    # 测试所有支持的算法
    print("\n--- 所有算法支持测试 ---")
    base_stats = {'scaler': 'standard', 'outlier_ratio_iqr': 0.01, 'skewness': 0.5}
    supported_algorithms = list(selector.algorithm_requirements.keys())

    for algorithm in supported_algorithms[:8]:  # 测试前8个算法避免输出太长
        scaler_type, reason = selector.recommend_for_algorithm(algorithm, base_stats)
        if scaler_type is not None:
            print(f"  {algorithm:20} -> {scaler_type:10} | {reason}")


def test_smart_scaler_selector():
    """测试 SmartScalerSelector 的功能"""
    print("\n=== 测试 SmartScalerSelector ===")

    selector = SmartScalerSelector()

    # 创建各种类型的数据
    test_data = {
        '正态分布': pd.Series(np.random.normal(0, 1, 100)),
        '偏态分布': pd.Series(np.random.exponential(2, 100)),
        '高异常值': pd.Series(np.concatenate([np.random.normal(0, 1, 90), np.random.normal(10, 1, 10)])),
        '常数列': pd.Series([5, 5, 5, 5, 5]),
        '小样本': pd.Series([1, 2]),
        '均匀分布': pd.Series(np.random.uniform(0, 10, 100)),
    }

    print("单列数据分析结果:")
    for name, data in test_data.items():
        stats_info = selector.analyze_feature(data)
        if stats_info.get('recommendation') == 'skip':
            scaler_type = 'skip'
            reason = stats_info['reason']
        else:
            scaler_type, reason = selector.recommend_scaler(stats_info)

        print(f"  {name:10} -> {scaler_type:15} | 样本数: {stats_info.get('n_samples', 'N/A'):3} | 理由: {reason}")

    # 测试整个DataFrame的处理
    print("\n--- DataFrame处理测试 ---")
    X = pd.DataFrame({
        'normal_col': np.random.normal(0, 1, 50),
        'skewed_col': np.random.exponential(1, 50),
        'outlier_col': np.concatenate([np.random.normal(0, 1, 45), np.random.normal(8, 1, 5)]),
        'constant_col': np.ones(50),
        'small_col': [1, 2, 3, 4] + [5] * 46  # 前4个值不同，后面常数值
    })

    selector.process(X)
    recommendations = selector.get_recommendations()

    print("DataFrame处理结果:")
    for col, info in recommendations.items():
        print(f"  列 {col:15} -> {info['scaler']:15} | 理由: {info['reason']}")


def test_unified_feature_scaler():
    """测试 UnifiedFeatureScaler 的基本功能"""
    print("\n=== 测试 UnifiedFeatureScaler ===")

    # 创建测试数据
    np.random.seed(42)
    n_samples = 100

    # 创建包含不同特征类型的数据
    data = {
        'normal_feature': np.random.normal(0, 1, n_samples),  # 正态分布
        'skewed_feature': np.random.exponential(2, n_samples),  # 偏态分布
        'outlier_feature': np.concatenate([np.random.normal(0, 1, n_samples - 5),
                                           np.random.normal(10, 1, 5)]),  # 包含异常值
        'constant_feature': np.ones(n_samples),  # 常数列
        'small_range_feature': np.random.uniform(0, 1, n_samples),  # 小范围
        'large_range_feature': np.random.uniform(0, 1000, n_samples),  # 大范围
    }

    X = pd.DataFrame(data)
    y = np.random.randint(0, 2, n_samples)  # 随机标签

    print("原始数据统计:")
    print(X.describe())

    # 测试1: 默认配置
    print("\n--- 测试1: 默认配置 ---")
    scaler1 = UnifiedFeatureScaler(algorithm='random_forest')
    X_scaled1 = scaler1.fit_transform(X,y=None)

    print("默认配置标准化后数据统计:")
    print(X_scaled1.describe())

    # 获取标准化报告
    report1 = scaler1.get_scaling_report()
    print("\n标准化报告:")
    print(f"算法: {report1['algorithm']}")
    print(f"总特征数: {report1['summary']['total_features']}")
    for method, count in report1['summary'].items():
        if method.endswith('_count') and count > 0:
            print(f"{method}: {count}")

    # 测试2: 指定配置
    print("\n--- 测试2: 指定配置 ---")
    method_config = {
        'transformers': [
            ('minmax', {'columns': ['small_range_feature'], 'feature_range': (0, 1)}),
            ('standard', {'columns': ['normal_feature']}),
            ('robust', {'columns': ['outlier_feature'], 'quantile_range': (10, 90)})
        ],
        'skip_scale': ['constant_feature']
    }

    scaler2 = UnifiedFeatureScaler(method_config=method_config, algorithm='neural_network')
    X_scaled2 = scaler2.fit_transform(X,y=None)

    print("指定配置标准化后数据统计:")
    print(X_scaled2.describe())

    # 测试3: 不同算法的比较
    print("\n--- 测试3: 不同算法比较 ---")
    algorithms = ['neural_network', 'svm', 'kmeans', 'random_forest', 'linear_regression']

    for algo in algorithms:
        scaler_algo = UnifiedFeatureScaler(algorithm=algo)
        X_scaled_algo = scaler_algo.fit_transform(X,y=None)
        report_algo = scaler_algo.get_scaling_report()

        methods_used = []
        for col, config in scaler_algo.scaling_config.items():
            methods_used.append(config['method'])

        unique_methods = set(methods_used)
        print(f"算法 {algo:20} -> 使用的标准化方法: {unique_methods}")


def test_pipeline_integration():
    """测试 Pipeline 集成"""
    print("\n=== 测试 Pipeline 集成 ===")

    # 创建更真实的数据
    X, y = make_classification(
        n_samples=200,
        n_features=6,
        n_informative=4,
        n_redundant=2,
        random_state=42
    )

    X_df = pd.DataFrame(X, columns=[
        'feature_1', 'feature_2', 'feature_3',
        'feature_4', 'feature_5', 'feature_6'
    ])

    # 添加一些特性来测试标准化器的智能选择
    X_df['feature_1'] = X_df['feature_1'] * 1000  # 大范围
    X_df['feature_2'] = np.where(X_df['feature_2'] > 1, X_df['feature_2'] * 10, X_df['feature_2'])  # 异常值
    X_df['feature_3'] = np.exp(X_df['feature_3'])  # 偏态分布

    algorithms = ['neural_network', 'svm', 'random_forest']

    for algorithm in algorithms:
        print(f"\n--- 使用算法: {algorithm} ---")

        pipeline = Pipeline([
            ('scaler', UnifiedFeatureScaler(algorithm=algorithm)),
            ('classifier', RandomForestClassifier(n_estimators=20, random_state=42))
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=0.2, random_state=42
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"测试准确率: {accuracy:.4f}")

        # 显示标准化配置
        scaler = pipeline.named_steps['scaler']
        print("标准化方法分布:")
        method_count = {}
        for col, config in scaler.scaling_config.items():
            method = config['method']
            method_count[method] = method_count.get(method, 0) + 1

        for method, count in method_count.items():
            print(f"  {method}: {count}列")


def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")

    # 空数据测试
    empty_data = pd.DataFrame()
    scaler_empty = UnifiedFeatureScaler()
    try:
        X_empty_scaled = scaler_empty.fit_transform(empty_data,y=None)
        print("空数据处理完成")
    except Exception as e:
        print(f"空数据处理异常: {e}")

    # 单列数据测试
    single_col_data = pd.DataFrame({'single_col': [1, 2, 3, 4, 5]})
    scaler_single = UnifiedFeatureScaler()
    X_single_scaled = scaler_single.fit_transform(single_col_data,y=None)
    print(f"单列数据标准化完成: {X_single_scaled.shape}")

    # 包含NaN的数据测试
    nan_data = pd.DataFrame({
        'col_with_nan': [1, 2, np.nan, 4, 5],
        'col_all_nan': [np.nan, np.nan, np.nan,np.nan, np.nan],
        'col_normal': [1, 2, 3, 4, 5]
    })

    scaler_nan = UnifiedFeatureScaler()
    try:
        X_nan_scaled = scaler_nan.fit_transform(nan_data,y=None)
        print("包含NaN数据处理完成")
        print(f"NaN数据标准化配置: {scaler_nan.scaling_config}")
    except Exception as e:
        print(f"包含NaN数据处理异常: {e}")

    # 测试极端异常值
    print("\n--- 极端异常值测试 ---")
    extreme_outlier_data = pd.DataFrame({
        'extreme_col': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1000]  # 一个极端异常值
    })

    scaler_extreme = UnifiedFeatureScaler(algorithm='svm')
    X_extreme_scaled = scaler_extreme.fit_transform(extreme_outlier_data,y=None)
    print(f"极端异常值处理: {scaler_extreme.scaling_config['extreme_col']}")


def test_performance():
    """测试性能"""
    print("\n=== 性能测试 ===")

    # 创建大数据集
    n_samples = 1000
    n_features = 50

    X_large, y_large = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=20,
        n_redundant=10,
        random_state=42
    )

    X_large_df = pd.DataFrame(X_large, columns=[f'feature_{i}' for i in range(n_features)])

    import time

    start_time = time.time()
    scaler_large = UnifiedFeatureScaler(algorithm='neural_network')
    X_large_scaled = scaler_large.fit_transform(X_large_df,y=None)
    end_time = time.time()

    print(f"处理 {n_samples} 样本 × {n_features} 特征耗时: {end_time - start_time:.4f} 秒")

    # 检查内存使用
    original_memory = X_large_df.memory_usage(deep=True).sum() / 1024  # KB
    scaled_memory = X_large_scaled.memory_usage(deep=True).sum() / 1024  # KB

    print(f"原始数据内存: {original_memory:.2f} KB")
    print(f"标准化后内存: {scaled_memory:.2f} KB")

    # 性能对比：不同算法的处理时间
    print("\n--- 不同算法性能对比 ---")
    algorithms = ['neural_network', 'svm', 'random_forest']

    for algorithm in algorithms:
        start_time = time.time()
        scaler = UnifiedFeatureScaler(algorithm=algorithm)
        _ = scaler.fit_transform(X_large_df.iloc[:500, :10],y=None)  # 使用子集测试
        end_time = time.time()
        print(f"算法 {algorithm:20} 处理时间: {end_time - start_time:.4f} 秒")


def test_inverse_transform_comprehensive():
    """测试全面的逆变换功能"""
    print("\n=== 测试逆变换功能 ===")

    # 创建包含各种类型的数据
    X = pd.DataFrame({
        'minmax_col': np.random.uniform(0, 100, 50),
        'standard_col': np.random.normal(0, 10, 50),
        'robust_col': np.concatenate([np.random.normal(0, 1, 45), np.random.normal(10, 1, 5)]),
        'constant_col': np.ones(50),
        'skewed_col': np.random.exponential(2, 50)
    })

    # 使用指定配置
    method_config = {
        'transformers': [
            ('minmax', {'columns': ['minmax_col'], 'feature_range': (0, 1)}),
            ('standard', {'columns': ['standard_col']}),
            ('robust', {'columns': ['robust_col']})
        ],
        'skip_scale': ['constant_col']
    }

    scaler = UnifiedFeatureScaler(method_config=method_config, algorithm='neural_network')
    X_scaled = scaler.fit_transform(X,y=None)

    # 逆变换
    X_restored = scaler.inverse_transform(X_scaled)

    print("逆变换准确性检查:")
    for col in X.columns:
        original_mean = X[col].mean()
        restored_mean = X_restored[col].mean()
        diff = abs(original_mean - restored_mean)
        diff_pct = (diff / abs(original_mean)) * 100 if original_mean != 0 else 0

        status = "✓" if diff < 1e-10 else "⚠" if diff_pct < 1 else "✗"
        print(f"  {status} 列 '{col:15}': 均值差异 = {diff:.8f} ({diff_pct:.4f}%)")

    # 测试部分列逆变换
    print("\n--- 部分列逆变换测试 ---")
    partial_cols = ['minmax_col', 'standard_col']
    X_partial_restored = scaler.inverse_transform(X_scaled, target_columns=partial_cols)

    for col in partial_cols:
        original_val = X[col].iloc[0]
        restored_val = X_partial_restored[col].iloc[0]
        diff = abs(original_val - restored_val)
        print(f"  列 '{col}': 第一个值 {original_val:.4f} -> 恢复后 {restored_val:.4f} | 差异: {diff:.8f}")


if __name__ == "__main__":
    # 运行所有测试
    test_algorithm_aware_selector()
    test_smart_scaler_selector()
    test_unified_feature_scaler()
    test_pipeline_integration()
    test_edge_cases()
    test_performance()
    test_inverse_transform_comprehensive()

    print("\n=== 所有测试完成 ===")