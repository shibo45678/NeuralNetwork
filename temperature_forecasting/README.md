# temperature_forecasting
Machine Learning Project


#    --> 如果有分类列 ， 自动配置 inputs - Embedding
#    --> 共同的模型构建
#    --> 输出层调整（每个输出1个层）损失函数等
#    self.model.fit() 训练模型（参数需要trainsets val_dataset）
#    逆转换回原数据 self._format_predictions(predictions) / 标签编码



ProcessTimeseriesColumns 类文档
主要功能：
智能时间列检测：自动识别数值型、字符串型、时间戳型时间列
时间类型转换：支持 Unix 时间戳、Excel 日期、字符串时间等格式转换
交互式选择：多时间列时让用户选择，字符串时间格式选择
特征工程：自动生成时间相关特征（季节、夜间标志、周期编码等）
数据验证：转换成功率检查，低成功率警告