# 完整的预处理流程
class CompletePreprocessor:
    """完整的预处理管道 - 统一处理 X 和 y"""

    def __init__(self):
        self.data_cleaners = []
        self.feature_engineers = []

    def add_data_cleaner(self, cleaner):
        self.data_cleaners.append(cleaner)

    def add_feature_engineer(self, engineer):
        self.feature_engineers.append(engineer)

    def fit_transform(self, X, y):
        """训练时的完整预处理"""
        X_temp, y_temp = X, y

        # 执行所有数据清洗步骤
        for cleaner in self.data_cleaners:
            X_temp, y_temp = cleaner.process(X_temp, y_temp)

        # 执行所有特征工程步骤
        for engineer in self.feature_engineers:
            X_temp, y_temp = engineer.fit_transform(X_temp, y_temp)

        return X_temp, y_temp

    def transform(self, X, y=None):
        """预测时的预处理"""
        # 处理不改变样本长度的预处理（标准化等）小类
        X_temp, y_temp = X, y

        for cleaner in self.data_cleaners:
            X_temp, y_temp = cleaner.process(X_temp, y_temp)

        for engineer in self.feature_engineers:
            X_temp, y_temp = engineer.transform(X_temp, y_temp)

        return X_temp, y_temp