from src.training.training_models import TrainingModel
from src.utils.windows import WindowGenerator
from src.data.processing import DataPreprocessor
import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import joblib
from typing import List,Dict



""" 训练模型重构"""

class ReconstructPredictor():

    def __init__(self):
        self.predict_data = None
        self.scalers=None
        self.constant_values=None
        self.reconstructed_model = None

    # 1. 加载训练产物

    # 2. 预处理新数据（标准化）
    def scale_data(self,scaler_path,constant_path,config:Dict,new_data:pd.DataFrame):
        """使用训练时scaler标准化/归一化"""
        self.scalers = joblib.load(scaler_path)
        self.constant_values = joblib.load(constant_path)
        self.scaler_config=config
        df=new_data.copy()

        for method,conf in config:
            print(f"处理 {method} 类型的列: {conf['columns']}")

            for col in conf['columns']:
                if col not in df.columns:
                    print(f"警告：{col}列不在预测数据列中，跳过该列标准化。预测数据与训练数据特征不匹配")
                    continue

                scaler =self.scalers[col]
                # 处理不同类型scaler
                if scaler is None:
                    print(f"列 {col}: 训练数据不足，无scaler，保持原样")
                    continue
                elif scaler =="constant":
                    constant_value =self.constant_values.get(key=col,default=0.5 if method == 'minmax' else 0)# 如果键不存在时返回的默认值
                    df[col]=constant_value
                    print(f"列 {col}: 常数列，设为{constant_value}")
                # 额外增加 iqr标准化里面的特别分支<=4，值不同的判断 ，不影响method遍历
                elif scaler['type'] =='manual_robust':
                    median = scaler['median']
                    std = scaler['scale']
                    df[col]=(df[col]-median)/std
                    print(f"列 {col}: 手动鲁棒标准化完成")
                else:# 正常应用transform 转换时的格式
                    try:
                        df[col] = scaler.transform(df[col].values.reshape(-1,1)).flatten()
                        print(f"列 {col}: {method}标准化完成")
                    except Exception as e:
                        print(f"列 {col} 标准化失败: {e}") # 失败时保持原样

        self.get_transform_stats()# 按方法类型统计
        self.validate_scalers()# 批量检查scaler状态

        return df

    # 3. 重建模型
    def reconstruct_trained_model(self,original_model,weights_path):
        """加载训练好的模型"""

        if hasattr(original_model, '_input_shape'):
            input_shape = original_model._input_shape
        else:
            raise ValueError("无法确定模型的输入形状")

        # 克隆模型结构
        best_model = tf.keras.models.clone_model(original_model)
        best_model.build((None,) + input_shape)  # 加上 batch

        # 重新编译（用于预测）
        best_model.compile(optimizer=original_model.optimizer,
                           loss=original_model.loss,
                           metrics=original_model.metrics)

        # 加载权重
        best_model.load_weights(weights_path)

        # 同样将形状加入到新模型
        best_model._input_shape = input_shape
        best_model._output_shape = getattr(original_model, '_output_shape', None)

        self.reconstructed_model = best_model

        return self

    # 4. 恢复建窗口数据并预测
    def predict_window_data(self,
                            origin_window:'WindowGenerator')->tf.data.Dataset:

        # 输入数据的最后时间点 + shift = 预测开始时间
        # 预测开始时间 + label_width = 预测结束时间
        new_window = WindowGenerator(
            input_width = origin_window.input_width,
            label_width = origin_window.label_width,
            shift = origin_window.shift,
            label_columns = origin_window.label_columns
        )
        try:
            if self.predict_data is not None and self.reconstructed_model is not None:
                new_input, _ = new_window.createDataset(self.predict_data) # 元组只需要inputs，不需要labels

                batch_predictions = self.reconstructed_model.predict(new_input)
                first_sample = batch_predictions[0]  # (5, 2)
                print(f"获取第一个样本的预测{first_sample}")
                first_timestep = batch_predictions[0, 0]  # [T_pred, p_pred]
                print(f"第一个时间步的T和p预测{first_timestep}")
                return batch_predictions
            else:
                print('预测数据未标准化/归一化 或者 模型未重建')

        except Exception as e:
            print(f"{str(e)}")


    def get_constr_model(self):
        if self.reconstructed_model  is not None:
            return self.reconstructed_model

    # 按方法类型统计
    def get_transform_stats(self):
        """获取各标准化方法的统计信息"""
        stats = {}
        for method, config in self.scaler_config:
            stats[method] = {
                'columns': config['columns'],
                'count': len(config['columns']),
                'scaler_types': {}
            }
            for col in config['columns']:
                scaler_type = type(self.scalers[col]).__name__ if hasattr(self.scalers[col], '__class__') else str(
                    self.scalers[col])
                stats[method]['scaler_types'][col] = scaler_type
        return stats

    # 批量检查scaler状态
    def validate_scalers(self):
        """验证所有scaler的状态"""
        for method, config in self.scaler_config:
            print(f"\n检查 {method} 方法的scaler:")
            for col in config['columns']:
                if col in self.scalers:
                    status = "正常" if self.scalers[col] is not None else "缺失"
                    print(f"  {col}: {status}")
                else:
                    print(f"  {col}: 未找到scaler")


    # 5. 反归一化到原始尺度(逆转换)
    def  _inverse_transform(self,predictions:pd.DataFrame,
                            target_columns:List=None)->pd.DataFrame:
        """将预测结果反归一化到原始尺寸"""

        if not hasattr(self,'scalers') or self.scalers is None or self.scaler_config is None :
            raise ValueError("没有找到scaler/scaler_config信息，请先call :scale_data()进行加载")

        df = predictions.copy()

        # 分组处理同类型scaler的列
        for method ,conf in self.scaler_config:
            print(f"处理 {method} 类型的列的逆标准化: {conf['columns']}")
            # 只处理目标列
            target_cols = [col for col in conf['columns'] if col in target_columns]
            if not target_cols:# 该方式下，列为空
                continue
            print(f"逆标准化 {method} 类型的目标列: {target_cols}")

            for col in target_cols:
                if col not in df.columns:
                    print(f"警告：{col}列不在待转换数据列中，跳过该列逆标准化。")
                    continue

                scaler = self.scalers[col]

                if scaler is None:  # 常数列
                    print(f"列 {col}: 无scaler，保持原样")
                    continue
                elif  scaler == 'constant': # 未标准化的列
                    original_value =self.constant_values.get(col) # get,比直接索引好，没有返回None,不会报错('minmax':值相同 col_max,'std':值相同 mean)
                    if  original_value is not None:
                        df[col] =original_value
                        print(f"列 {col}: 常数列逆标准化为{original_value}")

                elif isinstance(scaler,dict) and scaler.get('type') == 'manual_robust':
                    median= scaler['col_median']
                    std=scaler['scale'] # iqr换std
                    df[col]= df[col]*std+median
                    print(f"列 {col}: 手动鲁棒逆标准化完成")
                else:
                    try : # 逆标准化transform的数据格式和标准化一致
                        df[col]=scaler.inverse_transform(df[col].values.reshape(-1, 1)).flatten()
                        print(f"列 {col}: {method}逆标准化完成")
                    except Exception as e:
                        print(f"{col}列scaler逆标准化无效，可能没有inverse_transform方法{str(e)}")

        return   df














