import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from data.data_preprocessing.data_sampling import SystematicResampler


class TestResamplerIntegration:
    """采样器集成测试"""

    def test_with_sklearn_components_separately(self):
        """测试与 sklearn 组件分开使用（推荐方式）"""
        # 生成样本数据
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])
        y_series = pd.Series(y)

        # 1. 先进行采样
        sampler = SystematicResampler(start_index=0, step=2)
        X_sampled, y_sampled = sampler.learn_process(X_df, y_series)

        # 2. 然后使用 sklearn Pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])

        # 3. 训练和预测
        pipeline.fit(X_sampled, y_sampled)
        predictions = pipeline.predict(X_sampled)

        # 验证流程正常工作
        assert len(predictions) == len(y_sampled)
        assert hasattr(pipeline, 'predict')

    """时间序列：一般划分数据集不随机，且y不能走时间采样resample成数值型，使用上面的系统采样最简单"""
    # def test_time_based_with_ml_workflow(self):
    #     """测试时间重采样与机器学习工作流的集成"""
    #     # 创建时间序列数据
    #     dates = pd.date_range('2023-01-01', periods=48, freq='H')
    #     X = pd.DataFrame({
    #         'timestamp': dates,
    #         'feature1': np.random.randn(48),
    #         'feature2': np.random.randn(48)
    #     })
    #     y = pd.Series(np.random.randint(0, 2, 48)) # 0，1分类数据
    #
    #     # 时间重采样
    #     time_sampler = TimeBasedResampler(
    #         time_column='timestamp',
    #         freq='6H',  # 6小时间隔
    #         aggregation='mean'
    #     )
    #     X_sampled, y_sampled = time_sampler.learn_process(X, y)
    #
    #     # 创建机器学习模型
    #     from sklearn.ensemble import RandomForestClassifier
    #     from data.feature_engineering.split_datasets import train_val_test_split # 后续会改成适配sklearn模式
    #     # 拆分数据
    #     df = pd.concat([X_sampled,y_sampled],axis =1)
    #     res = train_val_test_split(df,train_size= 0.7, val_size = 0, test_size= 0.3) # 自定义拆分函数
    #     trainset = res[0]
    #     testset = res[2]
    #
    #     # 移除时间列用于建模
    #     X_for_model = trainset.drop(['timestamp','target'], axis=1)
    #
    #     y_for_model = trainset['target']
    #     X_test = testset.drop(['timestamp','target'],axis =1)
    #     y_test = testset['target']
    #
    #     # 训练模型
    #     model = RandomForestClassifier(n_estimators=10, random_state=42) # 时间序列 不随机
    #     model.fit(X_for_model, y_for_model)
    #
    #     # 验证模型工作
    #     score = model.score(X_test, y_test)
    #     assert 0 <= score <= 1  # 准确率应该在合理范围内

    def test_data_flow_consistency(self):
        """测试数据流的一致性"""
        # 创建有复杂索引的数据
        X = pd.DataFrame({
            'feature1': range(20),
            'feature2': range(20, 40)
        }, index=[f"row_{i}" for i in range(20)])  # 字符串索引

        y = pd.Series(range(20), index=[f"row_{i}" for i in range(20)])

        # 系统抽样
        sampler = SystematicResampler(start_index=3, step=4)
        X_sampled, y_sampled = sampler.learn_process(X, y)

        # 验证数据一致性
        assert X_sampled.index.equals(y_sampled.index)
        assert len(X_sampled) == len(y_sampled)

        # 验证采样逻辑正确
        expected_count = len(range(3, 20, 4))
        assert len(X_sampled) == expected_count