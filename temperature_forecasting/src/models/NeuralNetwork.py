# 神经网络的特征选择可选跳过：pass through (类型）
# 1.分割数据集
# 2.标准化 / 编码 可识别数值/分类->模型内部时已经编码标准化 都变成数值型（除非接入处理编码和标准化的步骤）
# 3.窗口
# __init__(output_configs=None, model_type='cnn', 窗口类 input_width=24, label_width=1, shift=1,batch_size=32, 训练类的epochs=100

# ------------------------------------------__fit__------------------------------------------
# _create_window_data : 获取窗口处理后的 datasets（inputs和outputs）dataset = self.window_generator.createDataset( 整体数据 X_processed)
#  -> a. split-datasets 获取  trainsets 和 validsets
#  -> b.是否还要根据数值和分类型调整形状？窗口不变还是？不变还需不需要调整形状 self.output_configs （enbedding / 窗口形状）- > output_name..regression


# _build_multi_task_model 构建模型：
# a. 共有部分 def：自动配置Embedding（如果有分类的）
# b. if-else分支判断使用模型的类型：cnn/lstm/混合 hybrid_model CNN提取特征 + LSTM时序建模 self.model_cnn = self._build_multi_task_model(X_windowed.shape[1:])
#    --> 如果有分类列 ， 自动配置 inputs - Embedding
#    --> 共同的模型构建
#    --> 输出层调整（每个输出1个层）损失函数等

# 4. self.model.fit() 训练模型（参数需要trainsets valsets）

# return self

# ------------------------------------------__predict__------------------------------------------
# 1. preprocessor_.transform(X)：
# 数据准备transform 预处理和特征处理内容 ：self.preprocessor.transform(X)-> 新数据
# 2._create_prediction_windows ：
# 获取窗口数据 testsets / newdata : testdataset = self.window_generator.make_dataset(test)
# 3. 模型预测 predictions = self.model.predict(dataset)
# 4. 逆转换回原数据 self._format_predictions(predictions) / 标签编码

import os
import joblib
from models.cnn import EnhancedCnnModel
from models.lstm import EnhancedLstmModel
from training import TrainingModel
from data.windows import WindowGenerator
from evaluation.metrics import ModelEvaluation
from data.decorator import validate_input, validate_output
from typing import Optional, Dict, List, Tuple, Literal, Union
from pydantic import BaseModel, model_validator, Field, field_validator
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.keras.regularizers import l2
import logging

logger = logging.getLogger(__name__)


class OutputConfig(BaseModel):
    """数据填充方法的配置类型"""
    output_configs: Dict[str, Dict] = Field(default={},
                                            description="输出配置 {输出列: {type: regression/classification, ...}}")

    @field_validator('output_configs')
    def _validate_output_configs(cls, v):
        """只验证新增参数"""
        for output_name, config in v.items():
            if not isinstance(config, dict):
                raise ValueError(f"输出配置 '{output_name}' 必须是字典")
            if 'type' not in config:
                raise ValueError(f"输出配置 '{output_name}' 必须包含 'type' 字段")
            if config['type'] not in ['regression', 'classification']:
                raise ValueError(f"输出类型必须是 'regression' 或 'classification'")

        return v


# 注意时间等不处理的列在模型里面怎么弄的 非数值非分类
# # 模型参数
# cnn_model_config = {
#     self._init_params['cnn_model_config']
# }
#
# input_shape = model_config.input_shape
# output_shape = model_config.output_shape
# branch_filters = model_config.branch_filters
# branch_kernels = model_config.branch_kernels
# branch_dilation_rate = model_config.branch_dilation_rate
# activation = model_config.activation
#
# self._init_params['lstm_model_config']
# lstm_model_config={
# input_shape = model_config.input_shape
#         output_configs = model_config.output_configs
#         learning_rate = model_config.learning_rate
#         units = model_config.units  # len控制lstm的层数
#         return_sequences = model_config.return_sequences  # 是否只在最后一个时间步产生输出，对应LSTM层数
#
# }
# TrainingModel(model_name:str,
#
#                 epochs: int = 20, # 总轮数
#                 verbose: int = 2
# weights_path f"best_model_{model_name}_weights.h5"
# time_col_name

class TimeSeriesEstimator(BaseEstimator, RegressorMixin, ClassifierMixin):
    def __init__(self, output_configs, **kwargs):  # 标准化
        """
        参数说明：
        - output_configs: 输出配置字典(每个输出特征单独一层)
                output_configs = {
                    'temperature': {'type': 'regression', # 单变量回归
                                    'loss':'mse',
                                    'metrics':['mae'],
                                    'units': 1,  #  每个时间步预测n个特征
                                    },

                    'weather_metrics': {'type': 'regression', # 多变量回归：比如经度和纬度
                                        'loss':'mse',
                                        'metrics':['mae'],
                                        'units': 4,           # 每个时间步预测4个指标
                                        },

                    'event_occurrence': {'type': 'binary_classification', # 二分类
                                        'loss':'binary_crossentropy',
                                        'metrics':['accuracy'],
                                        'units': 1,
                                        },

                    'weather_type': {'type': 'classification', # 多分类
                                    'loss':'sparse_categorical_crossentropy',
                                    'metrics':['accuracy'],
                                    'num_classes': 3,
                                    },
                    }
        - 分割数据集参数：略
        - 标准化参数：略
        - 窗口参数:略
        - 训练参数: 略
        """

        output_config_obj = OutputConfig(output_configs=output_configs or {})  # 直接创建实例
        self.output_configs = output_config_obj.output_configs
        self._init_params = kwargs  # 其他参数验证放在各自类

        # 设置属性以便sklearn的get_params工作
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.window_generator_ = None
        self.model_ = None
        self.is_fitted_ = False
        self.numeric_columns_ = {}
        self.categorical_columns_ = {}
        self.embedding_info = {}
        self.val_dataset_ = pd.DataFrame()
        self.train_dataset_ = pd.DataFrame()

        self.weights_path = f"best_model_{self.model_name}_weights.h5"
        self.history = None

    def fit(self, X, y=None):  # train_X
        """
        fit方法 - 支持从外部传入预处理器
        Args:
            X: 输入数据
            y: 目标数据（可选，对于多任务学习，目标可能在X中）
            preprocessor: 外部预处理器，如果为None则自动创建
        """
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        # 1. 保存特征列信息 / 时间等 怎么处理
        self.feature_columns_num =
        self.feature_columns_cat =

        # # 2. 分割数据集
        # train_X, val_X, = self._split_dataset(X)
        #
        # # 3. 内置标准化 + 编码
        # train_X_scaled = self._fit_transform_scaler(train_X)
        # val_X_scaled = self.  # 应用transform
        # test_X_scaled =
        # # 目前顺序就是原来的顺序 并没有将数值和分类分开

        # 4. 创建窗口数据
        self.window_generator_ = self._create_window_generator()
        self.train_dataset_ = self.window_generator_.createDataset(train_X_scaled)  # inputs  / outputs
        self.val_dataset_ = self.window_generator_.createDataset(val_X_scaled)
        self.test_dataset_ = self.window_generator_.createDataset(test_X_scaled)

        # 5. 构建模型
        # 5.1 获得embedding_info
        self.embedding_info = EmbeddingConfig._get_embedding_info(self.train_dataset_,
                                                                  self.categorical_columns_,
                                                                  self._init_params['input_shape'])
        # 5.2 选择模型
        if self.model_type == 'cnn':
            cnn_model = EnhancedCnnModel()
            cnn_model._build_multi_modal_cnn_model(self._init_params['cnn_model_config'],  # 修改成整体的configs
                                                   self.numeric_columns_,
                                                   self.categoric_columns_,
                                                   self.embedding_info)
            self.model_ = cnn_model

        elif self.model_type == 'lstm':
            lstm_model = EnhancedLstmModel()
            lstm_model._build_multi_modal_lstm_model(
                self._init_params['lstm_model_config'],
                self.numeric_columns_,
                self.categoric_columns_,
                self.embedding_info
            )
            self.model_ = lstm_model

        # 6. 训练模型
        self.history_ = TrainingModel(model_name=self._init_params['model_name'],
                                      model=self.model_,
                                      trainset=self.train_dataset_,
                                      valset=self.val_dataset_,
                                      verbose=self._init_params['verbose'],
                                      epochs=self._init_params['epochs'],
                                      weights_path=self.weights_path)

        self.is_fitted_ = True

        return self

    def predict(self, X):

        check_is_fitted(self)

        # 确保使用最佳权重
        if hasattr(self, 'weights_path') and os.path.exists(self.weights_path):
            self.model_.load_weights(self.weights_path)

        X_ = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        # 1. 标准化 + 编码
        X_scaled = self.scaler_.tranform()

        # 2. 窗口化
        window_data = self.window_generator_.createDataset(X_scaled)

        predictions = self.model_.predict(window_data)

        # 处理时间列
        historical_timestamps = X_[self._init_params['time_col_name']].copy()

        last_time = historical_timestamps.iloc[-1]
        steps_ahead = self._init_params['label_width']  # 默认预测步长

        future_timestamps = self._generate_futrue_timestamps(last_time, self._init_params['label_width'], 'H')

        predictions_ = pd.DataFrame({
            'timestamp': future_timestamps,
            'prediction': predictions.flatten()[:steps_ahead]  # # 确保长度匹配
        })

        return self._format_predicitons(predictions_)  # 整数编码的回溯

    def _create_window_generator(self):
        """创建窗口生成器 - 基于你的现有WindowGenerator"""

        window = WindowGenerator(
            input_width=self._init_params['input_width'],
            label_width=self._init_params['label_width'],
            shift=self._init_params['shift'],
            label_columns=list(self.output_configs.keys())
        )

        return window

    def evaluate(self):
        metrics = ModelEvaluation(self.output_configs, model_name=self._init_params['model_name'])
        metrics.comprehensive_model_evaluation(model=self.model_,
                                               window=self.window_generator_,
                                               valsets=self.val_dataset_,
                                               testsets=self.test_dataset_)

    def reconstruct_model(self):
        """加载训练好的模型"""

        if not hasattr(self, 'weights_path') or not os.path.exists(self.weights_path):
            raise ValueError('权重文件不存在，请先训练模型')

        if hasattr(self.model_, '_input_shape'):
            input_shape = self.model_._input_shape
        else:
            input_shape = self.model_.input_shape[1:]  # 去掉batch维度

        # 克隆模型结构
        reconstructed_model = tf.keras.models.clone_model(self.model_)
        reconstructed_model.build((None,) + input_shape)  # 加上 batch

        # 重新编译（用于预测）
        self._compile_model(reconstructed_model)

        # 加载权重
        reconstructed_model.load_weights(self.weights_path)

        return reconstructed_model

    def _generate_futrue_timestamps(self, last_time, n_steps, freq):
        return pd.date_range(start=last_time + self._init_params['shift'], periods=n_steps, freq=6 * freq)

    def _format_predicitons(self, data):

    # 分类列
    # 时间列

    def _compile_model(self, model):
        if hasattr(self, 'model_') and hasattr(self.model_, 'optimizer'):
            optimizer = self.model_.optimizer
        else:
            optimizer = 'adam'

        # 单输出或者多输出都可以使用字典，但是要保证输出层名字正确
        loss_config = self._get_loss_config()
        metrics_config = self._get_metrics_config()

        model.compile(
            optimizer=optimizer,
            loss=loss_config,  # 字典 键是输出层名
            metrics=metrics_config
        )

    def _get_loss_config(self):
        if not hasattr(self, 'output_configs') or not self.output_configs:
            # 默认配置
            loss = 'mse'
        else:
            loss = {}
            for output_name, config in self.output_configs.items():
                loss[f'output_{output_name}'] = config.get('loss', self._get_default_loss(config['type']))

        return loss

    def _get_metrics_config(self):
        """统一的metrics配置"""
        if not hasattr(self, 'output_configs') or not self.output_configs:
            metrics = ['mae']  # 默认

        metrics = {}
        for output_name, config in self.output_configs.items():
            metrics[f'output_{output_name}'] = config.get('metrics', self._get_default_metrics(config['type']))

        return metrics

    def _get_default_loss(self, type):
        return {'regression': 'mse',
                'classification': 'sparse_categorical_crossentropy',
                'binary_classification': 'binary_crossentropy'}[type]

    def _get_default_metrics(self, type):
        return {
            'regression': ['mae', 'mse'],
            'classification': ['accuracy'],
            'binary_classification': ['accuracy']
        }[type]

    def save(self, filepath):  # .pkl
        check_is_fitted(self)
        joblib.dump(self, filepath)
        print(f"完整模型已保存到: {filepath}")

    @classmethod
    def load(cls, filepath):
        return joblib.load(filepath)


class EmbeddingConfig:
    """Embedding维度选择配置"""

    @staticmethod
    def get_embedding_dim(n_categories: int) -> int:
        if n_categories <= 2:
            return 1  # 二分类
        elif n_categories <= 5:
            return 3  # 小类别
        elif n_categories <= 10:
            return 4  # 中等类别
        elif n_categories <= 20:
            return 6  # 较大类别
        elif n_categories <= 50:
            return 8  # 大类别
        else:
            # 谷歌研究公式：1.6 * n_categories^0.56
            return min(50, int(1.6 * n_categories ** 0.56))

    @staticmethod
    def should_use_embedding(n_categories: int, unique_ratio: float) -> bool:
        """
        判断是否应该使用Embedding
        unique_ratio: 唯一值数量 / 总样本数
        """
        # 高基数或中等基数都推荐使用Embedding
        return n_categories >= 2 and unique_ratio < 0.5

    @staticmethod
    def _get_embedding_info(dataset: pd.DataFrame, cat_cols: list, input_shape: tuple):
        embedding_configs = {}

        if cat_cols and isinstance(dataset, pd.DataFrame):

            for col in cat_cols:
                series = dataset[col].dropna()
                n_categories = series.nunique()
                unique_ratio = n_categories / len(series)

                base_config = {
                    'input_dim': n_categories,
                    'input_length': input_shape[0],
                    'name': f'embedding_{col}'
                }

                if EmbeddingConfig.should_use_embedding(n_categories, unique_ratio):
                    base_config['output_dim'] = EmbeddingConfig.get_embedding_dim(n_categories)
                else:  # 轻量Embedding
                    base_config.update({
                        'output_dim': max(1, min(2, n_categories // 20)),
                        'embeddings_regularizer': l2(0.1)  # 强正则化
                    })
                embedding_configs[col] = base_config

            return embedding_configs

    # def _extract_preprocessor_info(self, X):
    #     """从预处理器提取完整的列信息"""
    #     self.numeric_columns_ = []
    #     self.categorical_columns_ = []
    #     self.column_order_ = []
    #     self.categorical_info_ = {}
    #
    #     # 获取原始特征名
    #     if hasattr(self.preprocessor_, 'feature_names_in_'):
    #         feature_names = self.preprocessor_.feature_names_in_
    #     else:
    #         # 降级方案：使用X的列名
    #         feature_names = X.columns.tolist() if hasattr(X, 'columns') else []
    #
    #     # 解析 ColumnTransformer
    #     if hasattr(self.preprocessor_, 'transformers'):
    #         for name, transformer, columns in self.preprocessor_.transformers:
    #             if name != 'remainder':
    #                 # 记录列信息
    #                 if 'num' in name:
    #                     self.numeric_columns_.extend(columns)
    #                 elif 'cat' in name:
    #                     self.categorical_columns_.extend(columns)
    #
    #                 self.column_order_.extend(columns)
    #
    #                 # 提取分类信息
    #                 if 'cat' in name and hasattr(transformer, 'classes_'):
    #                     for col in columns:
    #                         self.categorical_info_[col] = {
    #                             'num_categories': len(transformer.classes_),
    #                             'classes': transformer.classes_
    #                         }
    #     else:
    #         # 如果不是ColumnTransformer，降级处理
    #         self._fallback_column_detection(X)

    # def _build_model(self, input_shape, numeric_columns, categorical_columns, categorical_info):
    #     inputs = tf.keras.layers.Input(shape=input_shape)
    #
    #     # 在模型内部拆分数值和分类特征
    #     if numeric_columns and categorical_columns:
    #         # 动态计算索引
    #         numeric_indices = [i for i, col in enumerate(self.all_columns)
    #                            if col in numeric_columns]
    #         categorical_indices = [i for i, col in enumerate(self.all_columns)
    #                                if col in categorical_columns]
    #
    #         numeric_features = tf.gather(inputs, numeric_indices, axis=-1)
    #         categorical_features = tf.gather(inputs, categorical_indices, axis=-1)
    #
    #         # 处理分类特征
    #         embedded_features = []
    #         for i, (col_name, idx) in enumerate(zip(categorical_columns, categorical_indices)):
    #             # 从categorical_features中提取单列
    #             single_cat = categorical_features[:, :, i:i + 1]
    #             single_cat_flat = tf.keras.layers.Reshape((input_shape[0],))(single_cat)
    #
    #             col_info = categorical_info[col_name]
    #             embedding = tf.keras.layers.Embedding(
    #                 col_info['num_categories'],
    #                 col_info['embedding_dim'],
    #                 input_length=input_shape[0]
    #             )(single_cat_flat)
    #             embedded_features.append(embedding)
    #
    #         # 合并特征
    #         if embedded_features:
    #             all_embedded = tf.keras.layers.Concatenate(axis=-1)(embedded_features)
    #             combined = tf.keras.layers.Concatenate(axis=-1)([numeric_features, all_embedded])
    #         else:
    #             combined = numeric_features
    #     else:
    #         combined = inputs
