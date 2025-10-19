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



def unify_feature_scaling(self, transformers: List = None) -> 'DataPreprocessor':  # 即zscore（原值-均值）/ 标准差
    """统一标准化和归一化数值特征
    ['minmax', {'feature_range': (0, 1), 'threshold': 1.5, 'columns': ['T']}
             归一化 (Normalization)	(x - min) / (max - min) scaler = MinMaxScaler(feature_range=(0, 1))  # 可选-1，1
    'std_scaler':{'threshold':3,'columns':[]},
             标准化 zscore 3倍标准差 (x-mean)/std
    'robust_scaler': {'quantile_range':(25, 75), 'columns':[]}]
            鲁棒 X_scaled = (X - median) / IQR
    """
    if transformers is None:
        print("统一数据格式但未提供方法")
        return self

    self.scalers = {}  # 保存每个列独立的scaler
    self.constant_values = {}  # 保存常数列的原始值
    for method, config in transformers:
        if method == 'minmax':
            print("数据归一化(min_max)...")
            # Min-Max 归一化（使用训练集统计量）

            for col in config['columns']:
                if self.trainSets[col].notna().sum() <= 1:  # 数据要求至少2个
                    print(f"列{col}:训练集数据不足，跳过归一化")
                    self.scalers[col] = None

                else:
                    train_col = self.trainSets[col].dropna()
                    col_max = train_col.max()
                    col_min = train_col.min()

                    if col_max == col_min:  # 所有值相同的情况
                        self.trainSets[col] = 0.5
                        self.constant_values[col] = col_max

                        if self.has_validation_set() and col in self.valSets.columns:
                            self.valSets[col] = 0.5
                        if not self.testSets.empty and col in self.testSets.columns:
                            self.testSets[col] = 0.5
                        print(f"列 '{col}': 最大值等于最小值，设为0.5")
                        # 标记这个列不需要scaler
                        self.scalers[col] = 'constant'

                    else:
                        # 定义： 只在训练集上拟合（计算参数）
                        scaler = MinMaxScaler(feature_range=config['feature_range'])  # 可选-1，1
                        scaler.fit(train_col)
                        # 将拟合好的scaler存入字典，以列名为键
                        self.scalers[col] = scaler

                        # 注意 1.transform要求的输入格式，2.转换完赋回去
                        self.trainSets[col] = scaler.transform(self.trainSets[col].values.reshape(-1, 1)).flatten()
                        print(
                            f"训练集列{col}:min_max 归一化完成")  # 1列Series(n,)->2D array(n,)。转换完用flatten()  压回1维(n,)
                        if self.has_validation_set() and col in self.valSets.columns:
                            self.valSets[col] = scaler.transform(self.valSets[col].values.reshape(-1, 1)).flatten()
                            print(f"验证集列{col}:min_max 归一化完成")
                        if not self.testSets.empty and col in self.testSets.columns:
                            self.testSets[col] = scaler.transform(
                                self.testSets[col].values.reshape(-1, 1)).flatten()
                            print(f"测试集列{col}:min_max 归一化完成")

            self.history.append("归一化min_max完成")

        if method == 'std_scaler':
            # zscore: （value-均值）/ 标准差 3 (dropna)
            print("数据标准化(zscore)...")
            for col in config['columns']:
                if self.trainSets[col].notna().sum() <= 1:  # 数据不足,至少2个非空
                    print(f"列{col}:训练集数据不足，跳过标准化")
                    self.scalers[col] = None
                else:
                    train_col = self.trainSets[col].dropna()  # 自动处理
                    col_mean = train_col.mean()  # 防止数据泄漏 只用训练集的均值和标准差
                    col_std = train_col.std()

                    if col_std == 0:  # 处理零标准差情况
                        self.constant_values[col] = col_mean
                        print(f"列{col} 的均值为{col_mean}，标准差为0 ，设为常数0")
                        self.trainSets[col] = 0  # 所有值相同，设为0
                        if self.has_validation_set() and col in self.valSets.columns:
                            self.valSets[col] = 0
                        if not self.testSets.empty and col in self.testSets.columns:
                            self.testSets[col] = 0
                        self.scalers[col] = 'constant'

                    else:  # std_z > 1e-8 避免除零错误
                        scaler = StandardScaler()
                        scaler.fit(train_col)
                        self.scalers[col] = scaler

                        self.trainSets[col] = scaler.transform(self.trainSets[col].values.reshape(-1, 1)).flatten()
                        print(f"训练集列{col}:z_score 标准化完成")
                        if self.has_validation_set() and col in self.valSets.columns:
                            self.valSets[col] = scaler.transform(self.valSets[col].values.reshape(-1, 1)).flatten()
                            print(f"验证集列{col}:z_score 标准化完成")
                        if not self.testSets.empty and col in self.testSets.columns:
                            self.testSets[col] = scaler.transform(
                                self.testSets[col].values.reshape(-1, 1)).flatten()
                            print(f"测试集列{col}:z_score 标准化完成")
                        print(f"列 '{col}': 标准化完成 (均值: {col_mean:.2f}, 标准差: {col_std:.2f})")

            self.history.append("数值列标准化(zscore)")

        if method == 'robust_scaler':
            print("数据鲁棒标准化(robust)...")

            for col in config['columns']:
                train_non_null = self.trainSets[col].dropna()
                n_samples = len(train_non_null)

                # 样本数 ≤ 1
                if n_samples <= 1:  # 数据要求至少2个
                    print(f"列{col}:训练集数据不足（{n_samples}个样本），跳过标准化")
                    self.scalers[col] = None
                    continue

                # 样本数 2-4，四分位数计算不稳定(分为:值相同 + 值不同的情况)
                elif n_samples <= 4:  # 处理：使用 中位数和标准差 替代 中位数和IQR 公式：(X - median) / std
                    print(f"列{col}:样本数较少({n_samples}个)，使用中位数和标准差代替 IQR")
                    col_median = train_non_null.median()
                    col_std = train_non_null.std()

                    if col_std == 0:  # 所有值相同的情况:标准差为0，设为0，记录中位数
                        self.trainSets[col] = 0
                        self.constant_values[col] = col_median

                        if self.has_validation_set() and col in self.valSets.columns:
                            self.valSets[col] = 0
                        if not self.testSets.empty and col in self.testSets.columns:
                            self.testSets[col] = 0
                        print(f"列 '{col}': 标准差为0，设为0")
                        self.scalers[col] = 'constant'
                    else:  # 值不相同的情况，手动实现基于标准差的鲁棒缩放
                        self.trainSets[col] = (self.trainSets[col] - col_median) / col_std
                        print(f"训练集{col}：使用中位数和标准差进行鲁棒标准化")

                        # 保存scaler信息供验证集和测试集使用
                        self.scalers[col] = {
                            'type': 'manual_robust',
                            'median': col_median,
                            'scale': col_std
                        }
                        if self.has_validation_set() and col in self.valSets.columns:
                            self.valSets[col] = (self.valSets[col] - col_median) / col_std
                            print(f"验证集列{col}：使用中位数和标准差进行鲁棒标准化")
                        if not self.testSets.empty and col in self.testSets.columns:
                            self.testSets[col] = (self.testSets[col] - col_median) / col_std
                            print(f"测试集列{col}：使用中位数和标准差进行鲁棒标准化")

                else:  # 数据足够（有一个特殊IQR=0）
                    q1 = train_non_null.quantile(0.25)
                    q3 = train_non_null.quantile(0.75)
                    iqr = q3 - q1

                    if iqr == 0:  # 原公式分母为0
                        self.trainSets[col] = 0
                        self.constant_values[col] = train_non_null.median()
                        if self.has_validation_set() and col in self.valSets.columns:
                            self.valSets[col] = 0
                        if not self.testSets.empty and col in self.testSets.columns:
                            self.testSets[col] = 0
                        print(f"列 '{col}': IQR为0，设为0")
                        self.scalers[col] = 'constant'
                    else:
                        scaler = RobustScaler(
                            with_centering=True,  # 减去中位数
                            with_scaling=True,  # 除以IQR
                            quantile_range=(25, 75)  # scikit-learn 使用百分比形式
                        )
                        scaler.fit(train_non_null.values.reshape(-1, 1))
                        self.scalers[col] = scaler
                        self.trainSets[col] = scaler.transform(self.trainSets[col].values.reshape(-1, 1)).flatten()
                        print(
                            f"训练集列{col}: robust标准化完成 (median={scaler.center_[0]:.3f}, IQR={1 / scaler.scale_[0]:.3f})")
                        # 转换验证集和测试集
                        if self.has_validation_set() and col in self.valSets.columns:
                            self.valSets[col] = scaler.transform(
                                self.valSets[col].values.reshape(-1, 1)
                            ).flatten()
                            print(f"验证集列{col}: robust标准化完成")

                        if not self.testSets.empty and col in self.testSets.columns:
                            self.testSets[col] = scaler.transform(
                                self.testSets[col].values.reshape(-1, 1)
                            ).flatten()
                            print(f"测试集列{col}: robust标准化完成")

            self.history.append("鲁棒标准化robust完成")