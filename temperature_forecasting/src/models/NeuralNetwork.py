import os
import joblib
from pydantic.v1 import validate_arguments
from pydantic import Field
from sklearn.utils.validation import check_is_fitted
from data.decorator import validate_input
from models.cnn import EnhancedCnnModel
from models.lstm import EnhancedLstmModel
from training.training_models import TrainingModel
from data.windows import WindowGenerator
from evaluation.model_evaluation import ModelEvaluation
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import tensorflow as tf
import pandas as pd
from tensorflow.python.keras.regularizers import l2
import logging

logger = logging.getLogger(__name__)


class TimeSeriesEstimator(BaseEstimator, RegressorMixin, ClassifierMixin):
    @validate_arguments
    def __init__(self,
                 model_config: dict = Field(..., description="必须提供包括模型配置（output_config）在内的、窗口配置。")):
        """
        Parameters:
        -----------
        model_config : dict, optional
            模型配置，用于训练新模型。通过各自模型进行参数验证
        saved_model_path : str, optional
            已保存模型路径，如果提供则直接使用保存的模型
        """
        self.model_config = model_config or {}
        self.weights_dir = f"best_model_{self.model_config['model_type']}_weights"
        self.best_checkpoint = None

        self.training_model_ = None  # 训练过程中使用的模型（可能包含dropout等）
        self.prediction_model_ = None  # 专门用于预测的最佳模型（已加载最佳权重）
        self.is_fitted_ = False
        self.embedding_info_ = {}
        self.history_ = None
        self.train_window_data = None
        self.val_window_data = None
        self.test_window_data = None
        self.window = self._create_window_generator()

    def fit(self, X, y=None):
        # 写出数据源
        train_datasets = X['train_datasets']
        val_datasets = X['val_datasets']


        # 1.处理窗口数据
        train_window_data = self.window.createDataset(train_datasets)
        val_window_data = self.window.createDataset(val_datasets)

        # 2. 构建模型 （神经网络预处理已经返回了模型期望的正确格式，不copy）
        # 1.1 获得embedding_info
        self.embedding_info = EmbeddingConfig._get_embedding_info(train_datasets,  # 原始DF
                                                                  self.model_config['categorical_columns'],
                                                                  self.model_config['input_shape'])
        # 1.2 选择模型
        if self.model_config['model_type'].startswith('cnn'):
            cnn_model_config = {**self.model_config['cnn_model_config'],  # 解包
                                'embedding_configs': self.embedding_info}  # 追加
            cnn_model = EnhancedCnnModel()
            cnn_model._build_multi_modal_cnn_model(cnn_model_config),
            self.training_model_ = cnn_model

        elif self.model_config['model_type'].startswith('lstm'):
            lstm_model_config = {**self.model_config['lstm_model_config'],
                                 'embedding_configs': self.embedding_info}
            lstm_model = EnhancedLstmModel()
            lstm_model._build_multi_modal_lstm_model(lstm_model_config)
            self.training_model_ = lstm_model

        # 3. 训练模型
        # 确保目录存在(立即创建)
        os.makedirs(self.weights_dir, exist_ok=True)
        self.history_, best_checkpoint = TrainingModel(model_name=self.model_config['model_name'],
                                                       model=self.training_model_,
                                                       trainset=train_window_data,
                                                       valset=val_window_data,
                                                       verbose=self.model_config['verbose'],
                                                       epochs=self.model_config['epochs'],
                                                       weights_dir=self.weights_dir)  # 目录
        # 保存最佳检查点路径供后续使用
        self.best_checkpoint = best_checkpoint

        # 训练完成后，创建用于预测的模型
        self._prediction_model = self.reconstruct_model()

        # 4. 评估模型
        self.evaluate_model(dataset=val_window_data, dataset_type='val')

        self.is_fitted_ = True

        return self

    @validate_input(validate_y=False)
    def predict(self, X):
        check_is_fitted(self)

        X_ = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        # 1. 处理窗口数据
        predict_window_data = self.window.createDataset(X_)

        # 2. 重构模型
        if self.prediction_model_ is None:
            self._prediction_model = self.reconstruct_model()  # 确保使用最佳权重

        # 3. 模型预测
        predictions = self._prediction_model.predict(predict_window_data)

        # 4. 恢复未使用时间列
        historical_timestamps = X_[self.model_config['time_column']].copy()

        last_time = historical_timestamps.iloc[-1]
        steps_ahead = self.model_config['label_width']  # 默认预测步长

        future_timestamps = self._generate_future_timestamps(last_time, self.model_config['label_width'], 'H')

        predictions_ = pd.DataFrame({
            'timestamp': future_timestamps,
            'prediction': predictions.flatten()[:steps_ahead]  # 确保长度匹配
        })

        return predictions_

    def _create_window_generator(self):

        window = WindowGenerator(
            input_width=self.model_config['input_width'],
            label_width=self.model_config['label_width'],
            shift=self.model_config['shift'],
            label_columns=list(self.model_config['output_config'].keys())
        )

        return window

    def reconstruct_model(self):
        """重构用于预测的干净模型"""

        if not hasattr(self, 'best_checkpoint'):
            raise ValueError('未找到最佳模型检查点')  # 现在改为分片 / 训练里面也有

        if hasattr(self.training_model_, '_input_shape'):
            input_shape = self.training_model_._input_shape
        else:
            input_shape = self.training_model_.input_shape[1:]  # 去掉batch维度

        # 克隆模型结构
        reconstructed_model = tf.keras.models.clone_model(self.training_model_)
        reconstructed_model.build((None,) + input_shape)  # 加上 batch

        # 加载权重
        reconstructed_model.load_weights(self.best_checkpoint)

        # 重新编译（用于预测）
        self._compile_for_prediction_model(reconstructed_model)

        return reconstructed_model

    def evaluate_model(self, dataset, dataset_type='val'):
        """用任意数据评估已训练好的模型"""
        model = self.reconstruct_model()

        metrics = ModelEvaluation(self.model_config['output_config'], model_name=self.model_config['model_name'])
        metrics.comprehensive_model_evaluation(model=model,  # 评估 best_model
                                               window=self.window,
                                               dataset=dataset,
                                               dataset_type=dataset_type)

    def _generate_future_timestamps(self, last_time, n_steps, freq):
        return pd.date_range(start=last_time + self.model_config['shift'], periods=n_steps, freq=6 * freq)

    def _compile_for_prediction_model(self, model):  # 同一Python进程中直接获取实例。独立的演化路径
        if hasattr(self, 'training_model_') and hasattr(self.training_model_, 'optimizer'):
            optimizer = self.training_model_.optimizer  # 可以用实例，load可以用配置
        else:
            learning_rate = self.model_config.get('learning_rate', 0.001)  # 保证学习率一致
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # 单输出或者多输出都可以使用字典，但是要保证输出层名字正确
        loss_config = self._get_loss_config()
        metrics_config = self._get_metrics_config()

        model.compile(
            optimizer=optimizer,
            loss=loss_config,  # 字典 键是输出层名
            metrics=metrics_config
        )

    def _get_compile_config_for_save(self):  # 磁盘恢复传递字典get_config()
        if hasattr(self, 'training_model_') and self.training_model_.optimizer:
            optimizer_config = self.training_model_.optimizer.get_config()  # 使用 get_config() 获取可序列化的配置
        else:
            # 回退逻辑
            learning_rate = self.model_config.get('learning_rate', 0.001)
            default_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            optimizer_config = default_optimizer.get_config()

        return {
            'optimizer': optimizer_config,
            'loss': self._get_loss_config(),
            'metrics': self._get_metrics_config()
        }

    def _get_loss_config(self):
        if not hasattr(self, 'output_config') or not self.model_config['output_config']:
            loss = 'mse'  # 重构在训练之后，训练已经检查output_config 不为空
        else:
            loss = {}
            for output_name, config in self.model_config['output_config'].items():
                loss[f'output_{output_name}'] = config.get('loss', self._get_default_loss(config['type']))

        return loss

    def _get_metrics_config(self):
        """统一的metrics配置"""
        if not hasattr(self, 'output_config') or not self.model_config['output_config']:
            metrics = ['mae']  # 重构在训练之后，训练已经检查output_config 不为空

        metrics = {}
        for output_name, config in self.model_config['output_config'].items():
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

    def clear_prediction_cache(self):
        """清空预测缓存"""
        if hasattr(self, '_prediction_model'):
            del self._prediction_model

    def save(self, save_path):
        """保存整个模型（包括配置、窗口、权重、编译配置）"""
        check_is_fitted(self)

        os.makedirs(save_path, exist_ok=True)

        # 1. 保存模型权重 （TF格式，支持大文件）
        if not hasattr(self, '_prediction_model'):
            self._prediction_model = self.reconstruct_model()

        # 使用TF格式保存权重（自动分片）
        weights_dir = os.path.join(save_path, 'model_weights')  # 文件夹放很很多文件
        self._prediction_model.save_weights(weights_dir, save_format='tf')

        # 2. 保存架构为Json
        model_json = self._prediction_model.to_json()
        with open(os.path.join(save_path, 'model_architecture.json'), 'w') as f:
            f.write(model_json)

        # 3. 保存配置信息
        save_configs = {
            'model_config': self.model_config,
            'window_config': {
                'input_width': self.window.input_width,
                'label_width': self.window.label_width,
                'shift': self.window.shift,
                'label_columns': self.window.label_columns},
            'compile_config': self._get_compile_config_for_save(),
        }

        joblib.dump(save_configs, os.path.join(save_path, 'saved_configs.pkl'))
        print(f"完整模型已保存到: {save_path}")
        return save_path

    @classmethod
    def load(cls, save_path):
        """加载分片保存的模型"""

        # 1. 加载配置
        config_path = os.path.join(save_path, 'saved_configs.pkl')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        config_data = joblib.load(config_path)

        # 2. 创建estimator实例
        estimator = cls(model_config=config_data['model_config'])

        # 3. 重建窗口生成器
        estimator.window = WindowGenerator(**config_data['window_config'])

        # 4. 从JSON重建模型结构
        model_json_path = os.path.join(save_path, 'model_architecture.json')
        if not os.path.exists(model_json_path):
            raise FileNotFoundError(f"模型架构文件不存在: {model_json_path}")

        with open(model_json_path, 'r') as f:
            model_json = f.read()

        # 处理自定义层(这里没有)
        custom_objects = getattr(cls, 'custom_objects', {})
        estimator.prediction_model_ = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)

        # 5. 加载分片权重
        weights_dir = os.path.join(save_path, 'model_weights')
        if not os.path.exists(weights_dir):
            raise FileNotFoundError(f"权重文件不存在: {weights_dir}")
        # 自动加载所有分片
        estimator.prediction_model_.load_weights(weights_dir)

        # 6. 重新编译模型 （调用 model.evaluate()，需要编译信息/保持与训练时行为一致）
        if 'compile_config' in config_data:
            estimator.prediction_model_.complie(**config_data['compile_config'])
        else:
            # 如果没有保存编译配置，使用默认编译
            estimator.prediction_model_.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )

        # 7. 标记为已拟合
        estimator.is_fitted_ = True

        # training_model_可以为None，因为不需要重新训练
        estimator.training_model_ = None

        print(f"模型已从 {save_path} 加载")
        return estimator

    def __getstate__(self):
        """序列化时只保留必要信息"""
        state = self.__dict__.copy()

        # 移除所有模型实例（通过save/load机制重建）
        state['training_model_'] = None
        state['prediction_model_'] = None
        state['window'] = None

        return state

    def __setstate__(self, state):
        """反序列化"""
        self.__dict__.update(state)

        if hasattr(self, 'weights_path') and os.path.exists(self.weights_path):
            self.prediction_model_ = self.reconstruct_model()
            self.window = self._create_window_generator()


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
                    'input_dim': n_categories,  # 不加1，因为已经预留了一个__UNKNOWN__
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
