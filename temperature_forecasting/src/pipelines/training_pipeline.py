class IndustrialMLSystem:
    def __init__(self):
        self.data_cleaner = DataCleaner()
        self.feature_engineer = FeatureEngineeringPipeline()
        self.model = Model() # 切分数据集 - 标准化/编码 - 窗口生成

    def train(self, raw_data, labels):
        print("开始训练工业级ML系统...")

        # 第1层：数据清洗（改变样本数量）
        clean_data, clean_labels = self.data_cleaner.process(raw_data, labels)
        print(f"数据清洗完成: {len(raw_data)} -> {len(clean_data)} 样本")

        # 第2层：特征工程（传入清洗后的 X 和 y）
        features, processed_labels = self.feature_engineer.fit_transform(clean_data, clean_labels)
        print(f"特征工程完成: {clean_data.shape[1]} -> {features.shape[1]} 特征")

        # 第3层：模型训练
        self.model.fit(features, processed_labels)
        print("模型训练完成")

        return self

    def predict(self, raw_data):
        print("进行预测...")

        # 应用相同的预处理流程
        clean_data, _ = self.data_cleaner.process(raw_data, None)
        features, _ = self.feature_engineer.transform(clean_data, None)
        predictions = self.model.predict(features)

        print(f"完成 {len(predictions)} 个样本的预测")
        return predictions


class FeatureEngineeringPipeline:
    """支持处理 X 和 y 的特征工程管道"""

    def __init__(self):
        self.engineers = []

    def add_engineer(self, engineer):
        self.engineers.append(engineer)

    def fit(self, X, y=None):
        X_temp, y_temp = X, y
        for engineer in self.engineers:
            engineer.fit(X_temp, y_temp)
            X_temp, y_temp = engineer.transform(X_temp, y_temp)
        return self

    def transform(self, X, y=None):
        X_temp, y_temp = X, y
        for engineer in self.engineers:
            X_temp, y_temp = engineer.transform(X_temp, y_temp)
        return X_temp, y_temp

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


# 使用示例
feature_pipeline = FeatureEngineeringPipeline()
feature_pipeline.add_engineer(SupervisedFeatureSelector(k=10))  # 需要 y
feature_pipeline.add_engineer(TargetEncodingEngineer())  # 需要 y

# 训练时：传入 X 和 y
features, labels = feature_pipeline.fit_transform(X_train, y_train)

# 预测时：可以只传入 X（y=None）
features_test, _ = feature_pipeline.transform(X_test, None)







