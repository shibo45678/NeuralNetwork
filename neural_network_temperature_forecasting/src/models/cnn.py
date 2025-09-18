from pydantic import BaseModel, Field, validator, PositiveInt
from dataclasses import dataclass,field
from typing import List, Optional, Dict, Tuple
import tensorflow as tf


# 基于历史六个时间点的天气情况（6行19列）预测经过24小时（shift=24)未来5个时间点的'T''p'（5*2）
class CnnModel:
    """"""
    """
    模型选择：1.适合时间序列回归预测(多个数值特征的回归)：short_sequence_model <20时间步、conv1D
           _build_sequential_model 方便扩展参数列表，支持更多类型的层，针对特定数据集定制最优架构；
                                   简单卷积+全连接
           _build_parallel_model 纯CNN模型，多分支设计可能捕捉更丰富的特征，比如短、中、长期
                                 层级优先，层内分支，合并，再送入下一层（inception风格）
                                  
            2.时间步级别的混合输出类型(数值回归+分类问题）
           _build_mixed_output_model 分类是已经预处理过的分类特征（分类数量较少），没涉及多分类密集向量embedding情况。
    
    """

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
        dense_units: List[PositiveInt] = Field(default=[5 * 19],
                                               description="全连接层单元数。值根据输出label形状定，如果只预测5行2列特征5*2；len控制全连接层层数。")
        learning_rate: float = Field(default=0.001, description="adam优化器学习率")

        @validator('padding')
        def validate_padding(cls, v):
            if v not in ['padding', 'valid']:
                raise ValueError("padding必须为'padding'或'valid'")
            return v

        @validator('learning_rate')
        def validate_learning_rate(cls, v):
            if not 0 <= v < 1:
                raise ValueError('learning_rate必须在[0, 1)范围内')
            return v

    # 配置类2：并行模型配置
    class ParallelConfig(BaseModel):  # 多分支、不使用全连接、回归预测
        input_shape: Tuple[PositiveInt, PositiveInt] = Field(default=(6, 19),
                                                             description="输入形状，例如 (6, 19) 表示6个时间步，19个特征")
        output_shape: Tuple[PositiveInt, PositiveInt] = Field(default=(5, 19),
                                                              description="输出形状。例如 (5,2) 表示预测5个时间步。每个时间步一个值, 2代表输出2个变量")
        branch_filters:List[List[PositiveInt]]=Field(default = [[32,32],[64,64]],description="每个子列表代表一个层级，子列表中的数字代表该层各个分支的滤波器数量，每个层级都是过滤器先小后大")
        branch_kernels:List[List[PositiveInt]]= Field(default=[[2, 3],[2,3]],
                                             description="每个子列表代表一个层级，子列表中的数字代表该层各个分支的kernel_size。控制短期特征、中期特征、长期特征")
        branch_dilation_rate:List[List[PositiveInt]] = Field(default=[[1, 1],[1,1]],
                                                      description="膨胀卷积，不增加参数的情况下扩大感受野，善于处理更长期的时间依赖。1是1D的默认值,(kernel_size-1)*dilation_rate+1=3, 1是默认值，长序列可调整")
        out_features: PositiveInt = Field(default=19, description="输出数值型的特征列数。")
    @validator('input_shape', 'output_shape')
    def validate_shape_length(cls, v):
        if len(v) != 2:
            raise ValueError(f"形状必须是长度为2的元组，当前长度为{len(v)}")
        return v

    @validator('out_shape')
    def validate_output_shape_consistency(cls, v, values):
        if 'out_features' in values and v[1] != values['out_features']:
            raise ValueError(
                f"输出形状的第二维 {v[1]} 必须与 out_features {values['out_features']} 保持一致"
            )
        return v

    @validator('branch_filters')
    def validate_branches(cls, v):
        for i, layer in enumerate(v):
            if len(layer) == 0:
                raise ValueError(f"第 {i} 层必须至少有一个分支")
        return v

    # 配置类3：高级模型配置
    class MixedConfig(BaseModel):
        input_shape: Tuple[PositiveInt, PositiveInt] = Field(default=(6, 19), description="输入形状")
        regression_features: PositiveInt = Field(default=1, description="输出数值型特征列个数")
        num_classes: PositiveInt = Field(default=3, description="输出分类型特征的列别数，如三分类012")
        output_timesteps: PositiveInt = Field(default=5, description="预测labels的时间步")

        @validator('input_shape')
        def validate_shape_length(cls, v):
            if len(v) != 2:
                raise ValueError(f"形状必须是长度为2的元组，当前长度为{len(v)}")
            return v

    def _validate_config(self, config: Optional[Dict], config_class: type) -> BaseModel:
        """通用的验证方法"""
        try:
            return config_class(**(config or {}))
        except Exception as e:
            raise ValueError(f"配置验证失败: {e}")

    def __init__(self,
                 architecture_type='parallel',  # 'sequential' / 'mixed'
                 **kwargs):
        if architecture_type == 'sequential':
            self.model = self._build_sequential_model(**kwargs)
        elif architecture_type == 'parallel':
            self.model = self._build_parallel_model(**kwargs)
        else:
            self.model = self._build_mixed_output_model(**kwargs)

    def _build_sequential_model(self,
                                config: dict = None
                                ) -> tf.keras.Sequential:

        model_config = self._validata_config(config, self.SequentialConfig)  # self.Seq

        filters = model_config.filters
        kernel_sizes = model_config.kernel_sizes
        strides = model_config.strides
        padding = model_config.padding
        activation = model_config.activation
        dense_units = model_config.dense_units
        learning_rate = model_config.learning_rate

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
        for i, units in enumerate(dense_units):
            # 全连接层会破坏位置信息，对时间序列不友好，短序列暂用。
            # 谨慎添加：kernel_initializer 仅用于数据已经标准化的初始化，该层权重矩阵全零，偏置也为0，寻求训练过程稳定性
            # 根据任务调整激活函数：分类、回归。softmax适用'多分类问题'的输出层，分类问题 Units 通常等于类别数量，
            model.add(tf.keras.layers.Dense(units=units,
                                            activation=activation[1],
                                            kernel_initializer=tf.initializers.zeros))
            print(f"添加全连接层dense_{i + 1}:Units={units}, activation ={activation}, 目前有全零初始化")

        # 添加输出层
        model.add(tf.keras.layers.Reshape([5, 19]))

        # 编译模型
        model.compile(
            optimizer=tf.optimizers.Adam(lr=learning_rate, epsilon=1e-07),  # adam 随机梯度下降
            loss=tf.losses.MeanSquaredError(),  # 损失函数 MSE
            metrics=[tf.metrics.MeanAbsoluteError()]  # 平均绝对值误差 MAE
        )

        return model

    def _build_parallel_model(self,
                              config=None
                              ) -> tf.keras.Model:

        model_config = self._validata_config(config, self.ParallelConfig)

        input_shape = model_config.input_shape
        branch_filters =model_config.branch_filters
        branch_kernels = model_config.branch_kernels
        branch_dilation_rate = model_config.branch_dilation_rate
        out_features = model_config.out_features

        #  多分支设计可能捕捉更丰富的特征，比如短、中、长期；
        self.inputs = tf.keras.Input(shape=input_shape)
        inputs=self.inputs.copy()
        # branch_filters: List[List[PositiveInt]] = Field(default=[[32, 32], [64, 64]],
        # branch_kernels: List[List[PositiveInt]] = Field(default=[[2, 3], [2, 3]],
        # 多分支特征提取
        for layer_index, (f_ls,k_ls, d_ls) in enumerate(zip(branch_filters,branch_kernels, branch_dilation_rate)):
            print(f"已添加第{layer_index}层")

            branch_outputs = []
            for branch_index,(num_filters,num_kernels,num_dilation) in enumerate(zip(f_ls,k_ls,d_ls)):
                branch = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=num_kernels, padding='same', dilation_rate=num_dilation)(inputs)
                print(f"第{branch_index}个分支:滤波器{num_filters}个，Kernel_size={num_kernels},Activation='relu',dilation_rate={num_dilation}")
                branch = tf.keras.layers.BatchNormalization()(branch)
                branch = tf.keras.layers.Activation('relu')(branch)

                branch_outputs.append(branch)

            # 层'间'的分支合并（融合同层不同分支的特征）
            merged = tf.keras.layers.concatenate(branch_outputs, axis=-1) # 拼接
            fused = tf.keras.layers.Conv1D(filters = sum(f_ls) // 2, kernel_size=1,padding='same', dilation_rate=1)(merged) # 合并后的大小还是32个特征?
            fused = tf.keras.layers.BatchNormalization()(fused)
            fused = tf.keras.layers.Activation('relu')(fused) # 要在上一步之后才能relu吗？

            inputs = fused # 进下一层前的数据
            self.outputs = fused # 也是本层输出


        # x = tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same', dilation_rate=1)(layer_outputs)
        # x = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same', dilation_rate=1)(x)

        # 对于时间序列预测，通常会在卷积层后直接使用卷积层或转置卷积层来调整输出形状，而不是全连接层 'linear'
        # outputs = tf.keras.layers.Conv1D(filters=out_features, kernel_size=1, activation=None,
        #                                  name='regression_output')(x)

        # 创建模型
        model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)

        # 编译模型
        model.compile(
            optimizer=tf.optimizers.Adam(lr=0.001, epsilon=1e-07),
            loss=tf.losses.MeanSquaredError(),
            metrics=[tf.metrics.MeanAbsoluteError()])

        return model

    def _build_mixed_output_model(self,
                                  config=None
                                  ) -> tf.keras.Model:
        """ 时间步级别的混合输出：每个时间步都预测数值和分类"""
        model_config = self._validata_config(config, self.MixedConfig)

        input_shape = model_config.input_shape
        regression_features = model_config.regression_features
        num_classes = model_config.num_classes
        output_timesteps = model_config.output_timesteps

        inputs = tf.keras.Input(shape=input_shape)

        # 共享特征提取（保持时间结构）
        x = tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(inputs)
        x = tf.keras.layers.Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv1D(512, kernel_size=3, activation='relu', padding='same')(x)

        # 并行输出分支（都保持时间维度）
        # 回归输出分支
        regression_branch = tf.keras.layers.Conv1D(64, kernel_size=1, activation='relu')(x)
        regression_output = tf.keras.layers.Conv1D(
            regression_features, kernel_size=1, activation=None, name='regression_output'
        )(regression_branch)

        # 分类输出分支（时间步级别的分类）
        classification_branch = tf.keras.layers.Conv1D(64, kernel_size=1, activation='relu')(x)
        classification_output = tf.keras.layers.Conv1D(
            num_classes, kernel_size=1, activation='softmax', name='classification_output'
        )(classification_branch)

        # 确保输出长度一致（如果需要可以裁剪）
        regression_output = regression_output[:, :output_timesteps, :]
        classification_output = classification_output[:, :output_timesteps, :]

        model = tf.keras.Model(inputs=inputs, outputs=[regression_output, classification_output])

        # 编译模型
        model.compile(
            optimizer='adam',
            loss={
                'regression_output': 'mse',
                'classification_output': 'categorical_crossentropy'
            },
            metrics={
                'regression_output': 'mae',
                'classification_output': 'accuracy'
            }
        )

        return model
