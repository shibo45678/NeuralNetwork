from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


def test_sklearn_compatible():
    """测试sklearn兼容版本"""
    print("测试 SklearnOutlierProcessor")
    print("=" * 50)

    # 创建测试数据
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.concatenate([np.random.normal(0, 1, 95), [10, -8, 12]]),
        'feature2': np.concatenate([np.random.normal(5, 2, 95), [20, -5, 25]])
    })
    y = pd.Series(np.concatenate([np.random.normal(10, 2, 95), [30, -10, 35]]))

    print(f"原始数据: X{X.shape}, y{len(y)}")

    # 方式1: 标记策略 - 适合pipeline
    print("\n1. 标记策略 (适合pipeline):")
    marker_pipeline = Pipeline([
        ('outlier_marker', SklearnOutlierProcessor(
            method_config={'method': 'iqr', 'threshold': 1.5},
            removal_strategy='mark'
        )),
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor())
    ])

    # 训练
    marker_pipeline.fit(X, y)
    print("✅ 标记策略pipeline训练成功")

    # 方式2: 独立使用移除策略
    print("\n2. 独立使用移除策略:")
    remover = SklearnOutlierProcessor(
        method_config={'method': 'iqr', 'threshold': 1.5},
        removal_strategy='remove'
    )
    X_clean, y_clean = remover.fit_transform(X, y)
    print(f"移除后数据: X{X_clean.shape}, y{len(y_clean)}")

    # 然后在清洗后的数据上构建pipeline
    clean_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor())
    ])
    clean_pipeline.fit(X_clean, y_clean)
    print("✅ 清洗后pipeline训练成功")

    # 获取报告
    report = remover.get_outlier_report()
    print(f"\n异常值报告: {report}")


if __name__ == "__main__":
    test_sklearn_compatible()