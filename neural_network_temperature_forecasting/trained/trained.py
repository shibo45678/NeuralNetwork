from src.training.training_models import TrainingModel
from src.utils.windows import WindowGenerator
from src.data.processing import DataPreprocessor
import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import joblib



""" 训练模型重构"""

class ReconstructPredictor():

    def __init__(self,
                 new_data,
                 scaler_path ,
                 model_weights_path,
                 original_model ):
        self.predict_data = new_data
        self.scaler = joblib.load(scaler_path)
        self.weights_path = model_weights_path
        self.original_model = original_model
        self.reconstructed_model = None


    # 1. 加载训练产物

    # 2. 预处理新数据（标准化）
    def scale_predict_data(self):
        """使用训练时scaler标准化/归一化"""
        df = self.predict_data.copy()
        try:
            for col, scaler in self.scaler.items():
                if col in df.columns:
                    df[col] = scaler.transform(df[col].values.reshape(-1, 1)).flatten()
                    print(f"{col}列完成标准化/归一化")
                else :
                    print(f"{col}列不在预测数据列中")
                    continue

            self.predict_data = df
            return self

        except Exception as e:
            print(f"使用训练时scaler标准化/归一化报错{str(e)}")
            return None



        # 获取最终统一数据用于预测
        part1_processed = predictor.get_data()
        pass

    # 3. 恢复建窗口数据

    #    artifacts =  joblib.load('prediction_artifacts.pkl') # 获取恢复窗口的部分
    #     predict_window = WindowGenerator(**window_config)
    #     model_input = predict_window.create_input_dataset(processed_data)


    def predict_window_data(self,
                            origin_window:'WindowGenerator',
                            new_cleaned_data:pd.DataFrame)->tf.data.Dataset:

        new_window = WindowGenerator(
            input_width = origin_window.input_width,
            label_width = origin_window.label_width,
            shift = origin_window.shift,
            label_columns = origin_window.label_columns,
            train_df = new_cleaned_data,
            val_df = pd.DataFrame(), # 空
            test_df = pd.DataFrame() # 空
        )

        new_input, _ = new_window.createTrainSet # 元组只需要inputs，不需要labels

        batch_predictions = self.reconstructed_model.predict(new_input)

        first_sample = batch_predictions[0]  # (5, 2)
        print(f"获取第一个样本的预测{first_sample}")
        first_timestep = batch_predictions[0, 0]  # [T_pred, p_pred]
        print(f"第一个时间步的T和p预测{first_timestep}")

        return batch_predictions






    def reconstruct_trained_model(self):
        """加载训练好的模型"""

        if hasattr(self.original_model, '_input_shape'):
            input_shape = self.original_model._input_shape
        else:
            raise ValueError("无法确定模型的输入形状")

        # 克隆模型结构
        best_model = tf.keras.models.clone_model(self.original_model)
        best_model.build((None,) + input_shape)  # 加上 batch

        # 重新编译（用于预测）
        best_model.compile(optimizer=self.original_model.optimizer,
                           loss=self.original_model.loss,
                           metrics=self.original_model.metrics)

        # 加载权重
        best_model.load_weights(self.weights_path)

        # 同样将形状加入到新模型
        best_model._input_shape = input_shape
        best_model._output_shape = getattr(self.original_model, '_output_shape', None)

        self.reconstructed_model = best_model

        return self


    def get_constr_model(self):
        if self.reconstructed_model  is not None:
            return self.reconstructed_model


    def  _inverse_transform(self, data:pd.DataFrame,
                            target_cols:List[str]=None,
                            method:str = 'denormalize'): # 'destandardize'
        """将预测结果反归一化到原始尺度"""
        if target_cols is None:
            columns =list(self.)






# 输入数据的最后时间点 + shift = 预测开始时间
# 预测开始时间 + label_width = 预测结束时间

# 预测脚本
# def predict(new_data, model_weights_path, artifacts_path):

#     # 1. 加载训练产物

#      artifacts =  joblib.load('prediction_artifacts.pkl') 还要有多个都在这里
#      需要包含：
#      feature_columns, （'prediction_artifacts.pkl'）具体哪些需要恢复的预处理的配置
#      scalers,
#      window_config,
#      model




#     # 4. 重新构建模型结构
#      artifacts =  joblib.load('prediction_artifacts.pkl') # 获取恢复模型的部分
#     model = create_model_structure()  # 您现有的模型创建函数
#
#
#
#
#
#
#
# 5. 预测
#     predictions = model.predict(model_input)  # 形状: (batch, 5, 2)
#
#
#   # # 6. 反归一化到原始尺度
# # predictions_original = preprocessor.inverse_transform(predictions_normalized, ['T', 'p'])
#
#     return predictions_original

# # 训练完成后分别保存
# joblib.dump(preprocessing_pipeline, 'preprocessor.pkl')  # 预处理管道
# model.save_weights('model_weights.h5')                   # 模型权重
# joblib.dump(window_config, 'window_config.pkl')          # 窗口配置
#
# # 预测时分别加载
# preprocessor = joblib.load('preprocessor.pkl')
# model.load_weights('model_weights.h5')
# window_config = joblib.load('window_config.pkl')



