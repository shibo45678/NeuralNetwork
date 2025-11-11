# temperature_forecasting
Machine Learning Project


1. class DataPreprocessor 只提供极端值方式，并不做处理
2. class ExtremeDataHandler 处理极端值（简单处理）




#    --> 如果有分类列 ， 自动配置 inputs - Embedding
#    --> 共同的模型构建
#    --> 输出层调整（每个输出1个层）损失函数等
#    self.model.fit() 训练模型（参数需要trainsets val_dataset）
#    逆转换回原数据 self._format_predictions(predictions) / 标签编码
