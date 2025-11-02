import tensorflow as tf
from pydantic import BaseModel, Field, PositiveInt
from typing import List, Optional, Dict, Tuple
from models.NeuralNetwork import EmbeddingConfig


class LstmModel:
    def __init__(self):
        self.model = None

    class SequentialConfig(BaseModel):
        units: List[PositiveInt] = Field(default=[64, ], description="滤波器数量，len控制lstm的层数")
        return_sequences: List[bool] = Field(default=[False, ],
                                             description="是否只在最后一个时间步产生输出，对应LSTM层数")
        output_shape: Tuple[PositiveInt, PositiveInt] = Field(default=(5, 2), description="调整reshape输出形状")

    def _validate_config(self, config: Optional[Dict], config_class: type) -> BaseModel:
        try:
            return config_class(**(config) or {})
        except Exception as e:
            raise ValueError(f"配置验证失败：{str(e)}")

    def _build_sequential_model(self,
                                config: dict = None
                                ) -> 'LstmModel':
        """
        如果单层效果不好：可以加LSTM层数
        units = [64, 32]  # 逐步压缩特征
        return_sequences = [True, False]
        如果效果还不好
        1. 更深的网络 [128, 64, 32]
        2. 更宽的网络 [64, 64]
        """

        """参数检查"""
        model_config = self._validate_config(config, self.SequentialConfig)

        units = model_config.units
        return_sequences = model_config.return_sequences
        output_shape = model_config.output_shape

        self.model = tf.keras.Sequential()

        # 添加LSTM层
        for i, (u, s) in enumerate(zip(units, return_sequences)):
            self.model.add(tf.keras.layers.LSTM(units=u, activation='tanh',
                                                return_sequences=s))  # shape[32,6,19]==>[32,64] tanh 将一个实数映射到（-1 1）的区间
            """
            1.设置只在最后一个时间步产生输出:return_sequences=false
            2.LSTM 层的参数总数【（64+19+1）*64】*4 == 【（上一轮输出+本轮输入）*（全联接输出）+（输出层偏置）】*4层（遗忘门*1+记忆门*2+输出门*1）
             -a. 如果LSTM是第1层，那么输入就是(64+inputs.shape[1])个特征值。
             -b. 如果是后续层，接在另一个LSTM层之后(且前一层的return_sequences=True),那么输入维度将是前一层的输出维度 64,总输入=64+64=128
            """

            print(f"添加LSTM层lstm_{i + 1}:Units={u},Activation='tanh',Return_sequences={s}")

        # 添加全连接层 (64+1)*95
        self.model.add(tf.keras.layers.Dense(units=output_shape[0] * output_shape[1],
                                             kernel_initializer=tf.initializers.zeros))  # dense  shape[32,95]
        print(f"添加Dense层:Units={output_shape[0] * output_shape[1]},设置全零初始化kernel_initializer")

        # 输出层,调整形状
        self.model.add(tf.keras.layers.Reshape(output_shape))  # [32,5,19]

        # 模型编译

        self.model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=0.001, epsilon=1e-07),
            loss=tf.losses.MeanSquaredError(),
            metrics=[tf.metrics.MeanAbsoluteError()])

        return self

    def summary(self):
        """委托给内部的 Keras 模型"""
        if self.model is not None:
            return self.model.summary()
        else:
            print("模型尚未构建")


class EnhancedLstmModel(LstmModel):
    class MultiModalConfig(LstmModel.SequentialConfig):
        input_shape: Tuple[int, int] = Field(default=(6, 19))
        numeric_columns: List[str] = Field(default=[])
        categorical_columns: List[str] = Field(default=[])
        embedding_configs: Dict[str, Dict] = Field(default={},
                                                   description="分类列信息 {input_dim,input_length,output_dim,embeddings_regularizer}}")
        output_configs: Dict[str, Dict] = Field(default={},
                                                description="输出配置 {输出列: {type: regression/classification, ...}}")
        learning_rate: float = Field(default=0.001)
        '''
        output_configs = {
            'temperature': {'type': 'regression', 'output_shape': (5, 1)},  # 5个时间步的回归 ,单特征输出 # 形状: (batch_size, 5, 1)
            'weather_type': {'type': 'classification', 'output_shape': (5, 1), 'num_classes': 3},
            'pressure': {'type': 'regression', 'output_shape': (5, 1)}
        }
        '''

    def _bulid_multi_modal_lstm_model(self, config: dict = None):
        model_config = self.MultiModalConfig(**(config or {}))

        input_shape = model_config.input_shape
        num_cols = model_config.numeric_columns
        cat_cols = model_config.categorical_columns
        embedding_configs = model_config.embedding_configs
        output_configs = model_config.output_configs
        learning_rate = model_config.learning_rate
        units = model_config.units  # len控制lstm的层数
        return_sequences = model_config.return_sequences  # 是否只在最后一个时间步产生输出，对应LSTM层数

        numeric_input = tf.keras.layers.Input(
            shape=(input_shape[0], len(num_cols)),
            name='numeric_input'
        )

        categorical_inputs = []
        for col_name in cat_cols:
            cat_input = tf.keras.layers.Input(
                shape=(input_shape[0],),  # (6, ) 表示6个时间步，1个特征
                name=f"categorical_{col_name}_input"
            )
            categorical_inputs.append(cat_input)

        # Embedding层处理分类特征
        embedded_layers = []
        for i, col_name in enumerate(cat_cols):
            embedding = tf.keras.layers.Embedding(**embedding_configs[col_name])(categorical_inputs[i])
            embedded_layers.append(embedding)
        if embedded_layers:
            if len(embedded_layers) > 1:
                all_embedded = tf.keras.layers.Concatenate(axis=-1)(embedded_layers)
            else:
                all_embedded = embedded_layers[0]
            combined = tf.keras.layers.Concatenate(axis=-1)([numeric_input, all_embedded])
        else:
            combined = numeric_input

        x = combined

        for i, (u, s) in enumerate(zip(units, return_sequences)):  # units列表长度代表 LSTM 层数
            x = tf.keras.layers.LSTM(units=u, activation='tanh', return_sequences=s, name=f'lstm_{i + 1}')(
                x)  # shape[32,6,19]==>[32,64] tanh 将一个实数映射到（-1 1）的区间
            x = tf.keras.layers.Dropout(0.2)(x)

        # 多任务输出
        outputs = []
        loss_dict = {}
        metric_dict = {}

        for output_name, output_config in output_configs.items():
            output_shape = output_config['output_shape']

            # 回归任务: 输出形状 (batch_size, 5, 1)
            if output_config['type'] == 'regression':
                output_layer = tf.keras.layers.Dense(output_shape[0] * output_shape[1],
                                                     name=f'dense_{output_name}')(x)
                output_layer = tf.keras.layers.Reshape(output_shape,
                                                       name=f'reshape_{output_name}')(output_layer)
                output_layer = tf.keras.layers.Activation('linear', name=f'activation_{output_name}')(output_layer)

                loss_dict[f'output_{output_name}'] = output_config.get('loss', 'mse')
                metric_dict[f'output_{output_name}'] = output_config.get('metrics', ['mae'])

            # 分类任务: 输出形状 (batch_size, 5, n_categories)
            elif output_config['type'] == 'classification':
                output_layer = tf.keras.layers.Dense(output_shape[0] * output_config['num_classes'],
                                                     name=f'dense_{output_name}')(x)
                output_layer = tf.keras.layers.Reshape((output_shape[0], output_config['num_classes']),
                                                       name=f'reshape_{output_name}')(output_layer)
                output_layer = tf.keras.layers.Activation('softmax', name=f'activation_{output_name}')(output_layer)

                loss_dict[f'output_{output_name}'] = output_config.get('loss', 'sparse_categorical_crossentropy')
                metric_dict[f'output_{output_name}'] = output_config.get('metrics', ['accuracy'])

            elif output_config['type'] == 'binary_classification':
                output_layer = tf.keras.layers.Dense(output_shape[0] * output_config['num_classes'],
                                                     name=f'dense_{output_name}')(x)
                output_layer = tf.keras.layers.Reshape((output_shape[0], output_config['num_classes']),
                                                       name=f'reshape_{output_name}')(output_layer)
                output_layer = tf.keras.layers.Activation('sigmoid', name=f'activation_{output_name}')(output_layer)
                loss_dict[f'output_{output_name}'] = output_config.get('loss', 'binary_crossentropy')
                metric_dict[f'output_{output_name}'] = output_config.get('metrics', ['accuracy'])

            else:
                output_layer = tf.keras.layers.Dense(output_shape[0] * output_shape[1],
                                                     name=f'dense_{output_name}')(x)
                output_layer = tf.keras.layers.Reshape(output_shape,
                                                       name=f'reshape_{output_name}')(output_layer)
                output_layer = tf.keras.layers.Activation('linear', name=f'activation_{output_name}')(output_layer)

                loss_dict[f'output_{output_name}'] = output_config.get('loss', 'mse')
                metric_dict[f'output_{output_name}'] = output_config.get('metrics', ['mae'])

            outputs.append(output_layer)

        all_inputs = [numeric_input] + categorical_inputs
        model = tf.keras.Model(inputs=all_inputs, outputs=outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_dict,
            metrics=metric_dict  # 简化处理
        )

        return model
