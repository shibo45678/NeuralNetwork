from pydantic import BaseModel, Field, field_validator, PositiveInt
from typing import List, Optional, Dict, Tuple
import os
import pandas as pd

# 在模型文件的开头设置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 最严格的设置
import tensorflow as tf


# 基于历史六个时间点的天气情况（6行19列）预测经过24小时（shift=24)未来5个时间点的'T''p'（5*2）
class CnnModel:
    """"""
    """
    模型选择：1.适合时间序列回归预测(仅多个数值特征的回归)：short_sequence_model <20时间步、conv1D
               _build_sequential_model 方便扩展参数列表，支持更多类型的层，针对特定数据集定制最优架构；
                                       简单卷积+全连接。
               _build_parallel_model 纯CNN模型，多分支设计可能捕捉更丰富的特征，比如短、中、长期。
                                     层级优先，层内分支，合并，再送入下一层（inception风格）。
                                     未考虑输出时间步大于输入时间步的情况。
                                      """

    def __init__(self,
                 architecture_type='parallel',  # 'sequential' / 'mixed'
                 **kwargs):
        if architecture_type == 'sequential':
            self.model = self._build_sequential_model(**kwargs)
        else:
            self.model = self._build_parallel_model(**kwargs)

    """=====================参数验证====================="""

    # 配置类1：_build_sequential_model 参数配置
    class SequentialConfig(BaseModel):  # 简单cnn + 全连接 ，回归预测
        filters: List[PositiveInt] = Field(default=[64, ],
                                           description="单层卷积层滤波器数量 / 特征通道(数)。len 控制卷积层层数")
        kernel_sizes: List[PositiveInt] = Field(default=[3, ],
                                                description="单层卷积核大小。只需提供'长'，1D'宽'自动适配窗口大小")
        strides: List[PositiveInt] = Field(default=[1, ],
                                           description="单层卷积步长/跨度。在输入数据上滑动的移动步数。步长越大：计算量越少｜保留信息越少｜成本越低")
        padding: List[str] = Field(default=['same', ],
                                   description="填充方式。'valid'不填充，序列长度减少，'same'填充，序列长度保持和inputs时间步一致。")
        activation: List[str] = Field(default=['relu', 'linear', 'sigmoid'],
                                      description="激活函数。单层的激活函数。代码中[0]用于卷积层，[1]用于全连接层。")
        output_shape: Tuple[PositiveInt, PositiveInt] = Field(default=(5, 19),
                                                              description="输出形状。例如 (5,2) 表示预测5个时间步。每个时间步一个值, 2代表输出2个变量")
        input_shape: Tuple[PositiveInt, PositiveInt] = Field(default=(6, 19), description="输入形状")
        learning_rate: float = Field(default=0.001, description="adam优化器学习率")

        @field_validator('padding')
        def validate_padding(cls, v):
            if v not in ['padding', 'valid']:
                raise ValueError("padding必须为'padding'或'valid'")
            return v

        @field_validator('learning_rate')
        def validate_learning_rate(cls, v):
            if not 0 <= v < 1:
                raise ValueError('learning_rate必须在[0, 1)范围内')
            return v

    # 配置类2：并行模型配置
    class ParallelConfig(BaseModel):  # 多分支、不使用全连接、回归预测 padding ='same' 保留长度 省略
        input_shape: Tuple[PositiveInt, PositiveInt] = Field(default=(6, 19),
                                                             description="输入形状，例如 (6, 19) 表示6个时间步，19个特征")
        output_shape: Tuple[PositiveInt, PositiveInt] = Field(default=(5, 19),
                                                              description="输出形状。例如 (5,2) 表示预测5个时间步。每个时间步一个值, 2代表输出2个变量")
        branch_filters: List[List[PositiveInt]] = Field(default=[[32, 32], [64, 64]],
                                                        description="每个子列表代表一个层级，子列表中的数字代表该层各个分支的滤波器数量，每个层级都是过滤器先小后大")
        branch_kernels: List[List[PositiveInt]] = Field(default=[[2, 3], [2, 3]],
                                                        description="每个子列表代表一个层级，子列表中的数字代表该层各个分支的kernel_size。控制短期特征、中期特征、长期特征")
        branch_dilation_rate: List[List[PositiveInt]] = Field(default=[[1, 1], [1, 1]],
                                                              description="膨胀卷积，不增加参数的情况下扩大感受野，善于处理更长期的时间依赖。1是1D的默认值,(kernel_size-1)*dilation_rate+1=3, 1是默认值，长序列可调整")
        activation: str = Field(default='relu')

    @field_validator('input_shape', 'output_shape')
    def validate_shape_length(cls, v):
        if len(v) != 2:
            raise ValueError(f"形状必须是长度为2的元组，当前长度为{len(v)}")
        return v

    @field_validator('branch_filters')
    def validate_branches(cls, v):
        for i, layer in enumerate(v):
            if len(layer) == 0:
                raise ValueError(f"第 {i} 层必须至少有一个分支")
        return v

    def _validate_config(self, config: Optional[Dict], config_class: type) -> BaseModel:
        """通用的验证方法"""
        try:
            return config_class(**(config or {}))
        except Exception as e:
            raise ValueError(f"配置验证失败: {e}")

    """===================构建模型==================="""

    def _build_sequential_model(self,
                                config: Optional[Dict] = None
                                ) -> 'tf.keras.Sequential':

        model_config = self._validate_config(config, self.SequentialConfig)  # self.Seq

        filters = model_config.filters
        kernel_sizes = model_config.kernel_sizes
        strides = model_config.strides
        padding = model_config.padding
        activation = model_config.activation
        output_shape = model_config.output_shape
        learning_rate = model_config.learning_rate
        input_shape = model_config.input_shape
        model = tf.keras.Sequential()

        # 添加卷积层
        for i, (f, k, s) in enumerate(zip(filters, kernel_sizes, strides)):
            model.add(tf.keras.layers.Conv1D  # 1d是单通道，适合处理文本时间序列，conv2d适合处理图像；
                      (filters=f, kernel_size=k, strides=1, padding=padding[0],
                       activation=activation[0],
                       name=f"conv_{i + 1}"))
            print(
                f"添加卷积层conv_{i + 1}:Filter={f},Kernel={k},Stride={s},Padding={padding[0]},Activation={activation[0]}")

        # 添加flatten层
        model.add(tf.keras.layers.Flatten())
        """不添加池化，预测是多个时间步，会丢失时间步。
        如果是单个时间步预测，使用GlobalAveragePooling1D()就不用flatten。
        因为池化会将每个分支的输出从3D张量（batch_size, timesteps, features）转换为2D张量（batch_size, features）"""

        # 添加全连接层
        for i, units in enumerate(output_shape[0] * output_shape[1]):
            # 全连接层会破坏位置信息，对时间序列不友好，短序列暂用。
            # 谨慎添加：kernel_initializer 仅用于数据已经标准化的初始化，该层权重矩阵全零，偏置也为0，寻求训练过程稳定性
            # 根据任务调整激活函数：分类、回归。softmax适用'多分类问题'的输出层，分类问题 Units 通常等于类别数量，
            model.add(tf.keras.layers.Dense(units=output_shape[0] * output_shape[1],
                                            activation=activation[1],
                                            kernel_initializer=tf.initializers.zeros))
            print(f"添加全连接层dense_{i + 1}:Units={units}, activation ={activation}, 目前有全零初始化")

        # 添加输出层
        model.add(tf.keras.layers.Reshape([output_shape[0], output_shape[1]]))

        # 编译模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-07),  # adam 随机梯度下降
            loss='mse',  # 损失函数 MSE
            metrics=['mae']  # 平均绝对值误差 MAE
        )

        # 保存input_shape到模型属性中
        model._input_shape = input_shape
        model._output_shape = output_shape

        return model

    def _build_parallel_model(self,
                              config: dict = None
                              ) -> tf.keras.Model:

        model_config = self._validate_config(config, self.ParallelConfig)

        input_shape = model_config.input_shape
        output_shape = model_config.output_shape
        branch_filters = model_config.branch_filters
        branch_kernels = model_config.branch_kernels
        branch_dilation_rate = model_config.branch_dilation_rate
        activation = model_config.activation

        #  多分支设计可能捕捉更丰富的特征，比如短、中、长期；

        inputs = tf.keras.Input(shape=input_shape)
        print(f"input_shape:{input_shape}")
        x = inputs

        # 多分支特征提取
        for layer_index, (f_ls, k_ls, d_ls) in enumerate(zip(branch_filters, branch_kernels, branch_dilation_rate)):
            print(f"已添加第{layer_index}层")

            branch_outputs = []
            for branch_index, (num_filters, num_kernels, num_dilation) in enumerate(zip(f_ls, k_ls, d_ls)):
                branch = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=num_kernels, padding='same',
                                                dilation_rate=num_dilation)(x)
                branch = tf.keras.layers.BatchNormalization()(branch)
                branch = tf.keras.layers.Activation(activation)(branch)
                print(
                    f"第{branch_index}个分支:滤波器{num_filters}个，Kernel_size={num_kernels},Activation={activation},dilation_rate={num_dilation}")
                branch_outputs.append(branch)

            # 层'内'的分支合并（融合同层不同分支的特征）将同层的多个分支沿着最后一个维度（即特征维度）拼接起来。
            # 2个(batch_size, time_steps, 32) ->(batch_size, time_steps, 64) 即:2个(batch_size,6,32)->(batch_size, 6, 64)
            merged = tf.keras.layers.concatenate(branch_outputs, axis=-1)  # 拼接
            print(f"已完成第{layer_index}层的分支合并")

            # 使用1×1卷积进行特征融合和降维
            fused = tf.keras.layers.Conv1D(filters=sum(f_ls) // 2, kernel_size=1, padding='same', dilation_rate=1)(
                merged)  # 降维到(各分支滤波器总数)的一半 [32，6，64]
            fused = tf.keras.layers.BatchNormalization()(fused)
            fused = tf.keras.layers.Activation(activation)(fused)

            # 残差连接：允许梯度直接从后期层流向早期层，缓解梯度消失问题
            if x.shape[-1] == fused.shape[-1]:  # 确保维度匹配
                x = tf.keras.layers.add([x, fused])
            else:
                # 如果维度不匹配，使用1*1 卷积调整
                shortcut = tf.keras.layers.Conv1D(filters=fused.shape[-1], kernel_size=1, padding='same',
                                                  dilation_rate=1)(x)
                x = tf.keras.layers.add([shortcut, fused])  # 向合并后的filter靠拢

        # 添加普通卷积层进一步融合特征
        x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation=activation, padding='same', dilation_rate=1)(x)

        # 使用注意力机制对时间步加权
        attention = tf.keras.layers.Conv1D(filters=1, kernel_size=1, activation='sigmoid', padding='same')(
            x)  # filter=1 所有特征共享权;每个特征独立权重 filter跟输出定；softmax(axis=-1)特征权重和为1	强调特征间相对重要性（竞争性）
        weighted = tf.keras.layers.multiply([x, attention])  # (batch_size,timesteps,64)

        # 保留时间维度的注意力机制。不用平均时间步 outputs = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(weighted)  # 形状: (batch_size, 64)
        x = weighted

        # 转置卷积-进行时间步后续的调整 (batch_size,timesteps,32) 与输入形状相同的时间步数,使输出长度 = 输入长度 × strides
        x = tf.keras.layers.Conv1DTranspose(filters=32, kernel_size=3, padding='same')(x)
        x = x[:, :output_shape[0], :]  # 裁剪到5个时间步 (32, 5, 2)

        # 使用1*1 卷积为每个时间步输出2个特征
        outputs = tf.keras.layers.Conv1D(filters=output_shape[1], kernel_size=1, padding='same')(x)

        # 创建模型
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model._input_shape = input_shape  # 保存到模型属性中
        model._output_shape = output_shape  # 方便训练调用

        # 编译模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-07),
            loss='mse',  # 损失函数 MSE
            metrics=['mae'])  # 平均绝对值误差 MAE

        return model

    def summary(self):
        """委托给内部的 Keras 模型"""
        if self.model is not None:
            return self.model.summary()
        else:
            print("模型尚未构建")


class EnhancedCnnModel(CnnModel):
    class MultiModalConfig(CnnModel.ParallelConfig):
        numeric_columns: List[str] = Field(default=[], description="数值列名称")
        categorical_columns: List[str] = Field(default=[], description="分类列名称")
        embedding_info: Dict[str, Dict] = Field(default={},
                                                description="分类列信息 {input_dim,input_length,output_dim,embeddings_regularizer}}")
        output_configs: Dict[str, Dict] = Field(default={},
                                                description="输出配置 {输出列: {type: regression/classification, ...}}")
        learning_rate: float = Field(default=0.001)
        '''
        参数说明：
        output_configs: 输出配置字典(每个输出特征单独一层)
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
        '''

        @field_validator('dataset')
        def validate_dataset(cls, v):
            if v is not None and not isinstance(v, pd.DataFrame):
                raise ValueError("必须是pandas DataFrame或None")
            return v

        @field_validator('output_configs')
        def _validate_output_configs(cls, v):
            for output_name, config in v.items():
                if not isinstance(config, dict):
                    raise ValueError(f"输出配置 '{output_name}' 必须是字典")
                if 'type' not in config:
                    raise ValueError(f"输出配置 '{output_name}' 必须包含 'type' 字段")
                if config['type'] not in ['regression', 'classification']:
                    raise ValueError(f"输出类型必须是 'regression' 或 'classification'")

            return v

    def _build_multi_modal_cnn_mode(self, config: dict = None) -> tf.keras.Model:
        model_config = self._validate_config(config, self.MultiModalConfig)

        input_shape = model_config.input_shape
        output_shape = model_config.output_shape
        branch_filters = model_config.branch_filters
        branch_kernels = model_config.branch_kernels
        branch_dilation_rate = model_config.branch_dilation_rate
        activation = model_config.activation
        num_cols = model_config.numeric_columns
        cat_cols = model_config.categorical_columns
        embedding_configs = model_config.embedding_configs
        output_configs = model_config.output_configs
        learning_rate = model_config.learning_rate

        # 分类列Embedding层的判断
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

        # 合并所有特征
        if embedded_layers:
            if len(embedded_layers) > 1:
                all_embedded = tf.keras.layers.Concatenate(axis=-1)(embedded_layers)
            else:
                all_embedded = embedded_layers[0]
            combined = tf.keras.layers.Concatenate(axis=-1)([numeric_input, all_embedded])
        else:
            combined = numeric_input

        x = combined

        # 多分支特征提取
        for layer_index, (f_ls, k_ls, d_ls) in enumerate(zip(branch_filters, branch_kernels, branch_dilation_rate)):
            print(f"已添加第{layer_index}层")

            branch_outputs = []
            for branch_index, (num_filters, num_kernels, num_dilation) in enumerate(zip(f_ls, k_ls, d_ls)):
                branch = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=num_kernels, padding='same',
                                                dilation_rate=num_dilation)(x)
                branch = tf.keras.layers.BatchNormalization()(branch)
                branch = tf.keras.layers.Activation(activation)(branch)
                print(
                    f"第{branch_index}个分支:滤波器{num_filters}个，Kernel_size={num_kernels},Activation={activation},dilation_rate={num_dilation}")
                branch_outputs.append(branch)

            # 层'内'的分支合并（融合同层不同分支的特征）将同层的多个分支沿着最后一个维度（即特征维度）拼接起来。
            # 2个(batch_size, time_steps, 32) ->(batch_size, time_steps, 64) 即:2个(batch_size,6,32)->(batch_size, 6, 64)
            merged = tf.keras.layers.concatenate(branch_outputs, axis=-1)  # 拼接
            print(f"已完成第{layer_index}层的分支合并")

            # 使用1×1卷积进行特征融合和降维
            fused = tf.keras.layers.Conv1D(filters=sum(f_ls) // 2, kernel_size=1, padding='same', dilation_rate=1)(
                merged)  # 降维到(各分支滤波器总数)的一半 [32，6，64]
            fused = tf.keras.layers.BatchNormalization()(fused)
            fused = tf.keras.layers.Activation(activation)(fused)

            # 残差连接：允许梯度直接从后期层流向早期层，缓解梯度消失问题
            if x.shape[-1] == fused.shape[-1]:  # 确保维度匹配
                x = tf.keras.layers.add([x, fused])
            else:
                # 如果维度不匹配，使用1*1 卷积调整
                shortcut = tf.keras.layers.Conv1D(filters=fused.shape[-1], kernel_size=1, padding='same',
                                                  dilation_rate=1)(x)
                x = tf.keras.layers.add([shortcut, fused])  # 向合并后的filter靠拢

        # 添加普通卷积层进一步融合特征
        x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation=activation, padding='same', dilation_rate=1)(x)

        # 使用注意力机制对时间步加权
        attention = tf.keras.layers.Conv1D(filters=1, kernel_size=1, activation='sigmoid', padding='same')(
            x)  # filter=1 所有特征共享权;每个特征独立权重 filter跟输出定；softmax(axis=-1)特征权重和为1	强调特征间相对重要性（竞争性）
        weighted = tf.keras.layers.multiply([x, attention])  # (batch_size,timesteps,64)

        # 保留时间维度的注意力机制。不用平均时间步 outputs = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(weighted)  # 形状: (batch_size, 64)
        x = weighted

        # 转置卷积-进行时间步后续的调整 (batch_size,timesteps,32) 与输入形状相同的时间步数,使输出长度 = 输入长度 × strides
        x = tf.keras.layers.Conv1DTranspose(filters=32, kernel_size=3, padding='same')(x)
        x = x[:, :output_shape[0], :]  # 裁剪到5个时间步 (32, 5, 2)

        # 多任务输出层
        outputs = []
        loss_dict = {}
        metric_dict = {}

        for output_name, output_config in output_configs.items():
            output_shape = output_config['output_shape']

            if output_config['type'] == 'regression':
                # 回归输出
                output_layer = tf.keras.layers.Conv1D(
                    filters=output_shape[1],  # 单特征输出
                    kernel_size=1,
                    activation='linear',
                    padding='same',
                    name=f'output_{output_name}'
                )(x)
                loss_dict[f'output_{output_name}'] = output_config.get('loss','mse')
                metric_dict[f'output_{output_name}'] = output_config.get('metrics',['mae'])

            elif output_config['type'] == 'classification':
                # 分类任务：输出 (batch_size, output_timesteps, num_classes)
                output_layer = tf.keras.layers.Conv1D(
                    filters=output_shape[1] * output_config['num_classes'],
                    kernel_size=1,
                    activation='softmax',
                    padding='same',
                    name=f'output_{output_name}'
                )(x)
                loss_dict[f'output_{output_name}'] = output_config.get('loss','sparse_categorical_crossentropy')
                metric_dict[f'output_{output_name}'] =output_config.get('metrics',['accuracy'])

            elif output_config['type'] == 'binary_classification':
                output_layer = tf.keras.layers.Conv1D(
                    filters=output_shape[1] * output_config['num_classes'],
                    kernel_size=1,
                    activation='sigmoid',
                    padding='same',
                    name=f'output_{output_name}'
                )(x)
                loss_dict[f'output_{output_name}'] = output_config.get('loss','binary_crossentropy')
                metric_dict[f'output_{output_name}'] = output_config.get('metrics',['accuracy'])

            else:
                # 默认情况：使用回归配置
                output_layer = tf.keras.layers.Conv1D(
                    filters=output_shape[1],
                    kernel_size=1,
                    activation='linear',
                    padding='same',
                    name=f'output_{output_name}'
                )(x)
                loss_dict[f'output_{output_name}'] = output_config.get('loss', 'mse')
                metric_dict[f'output_{output_name}'] = output_config.get('metrics', ['mae'])

            outputs.append(output_layer)

        all_inputs = [numeric_input] + categorical_inputs
        model = tf.keras.Model(inputs=all_inputs, outputs=outputs)

        # 不同的编译器
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=loss_dict,  # Keras通过输出层的name来匹配loss和metrics
                      metrics=metric_dict)

        return model


