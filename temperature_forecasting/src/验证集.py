from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. 创建Pipeline
pipeline = Pipeline([
    ('scaler', UnifiedFeatureScaler(transformers=transformers)),
    ('model', RandomForestRegressor())
])

# 2. 训练Pipeline
pipeline.fit(X_train, y_train)

# 获取转换器
scaler = pipeline.named_steps['scaler']
model = pipeline.named_steps['model']

y_pred = pipeline.predict(X_val)
# 5. 用转换后的数据重新训练（确保一致性）
model_on_transformed = RandomForestRegressor(**model.get_params())
model_on_transformed.fit(X_train_transformed, y_train)

# 6. 预测和评估
y_pred_pipeline = pipeline.predict(X_val)  # Pipeline自动转换
y_pred_manual = model_on_transformed.predict(X_val_transformed)  # 手动转换

# 7. 比较结果
mse_pipeline = mean_squared_error(y_val, y_pred_pipeline)
mse_manual = mean_squared_error(y_val, y_pred_manual)
r2_pipeline = r2_score(y_val, y_pred_pipeline)
r2_manual = r2_score(y_val, y_pred_manual)

print(f"Pipeline MSE: {mse_pipeline:.4f}")
print(f"手动转换 MSE: {mse_manual:.4f}")


class ModelTrainer:
    def __init__(self, transformers):
        self.transformers = transformers
        self.pipeline = None
        self.scaler = None
        self.model = None

    def train(self, X_train, y_train):
        """训练完整流程"""
        self.pipeline = Pipeline([
            ('scaler', UnifiedFeatureScaler(transformers=self.transformers)),
            ('model', RandomForestRegressor())
        ])

        # 训练
        self.pipeline.fit(X_train, y_train)

        # 保存组件
        self.scaler = self.pipeline.named_steps['scaler']
        self.model = self.pipeline.named_steps['model']

        return self

    def predict(self, X_new):
        """预测新数据"""
        if self.scaler is None or self.model is None:
            raise ValueError("请先调用train方法训练模型")

        # 用训练好的流程处理新数据
        X_transformed = self.scaler.transform(X_new)
        y_pred = self.model.predict(X_transformed)

        return y_pred, X_transformed

    def evaluate(self, X_val, y_val):
        """评估验证集"""
        y_pred, X_val_transformed = self.predict(X_val)

        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        # 返回详细结果，包括转换后的数据用于分析
        return {
            'predictions': y_pred,
            'X_transformed': X_val_transformed,
            'metrics': {'mse': mse, 'r2': r2}
        }


# 使用示例
trainer = ModelTrainer(transformers)
trainer.train(X_train, y_train)

# 评估验证集
results = trainer.evaluate(X_val, y_val)
print(f"验证集MSE: {results['metrics']['mse']:.4f}")

# 可以看到转换后的验证集数据
print("转换后的验证集:", results['X_transformed'].head())

