import tensorflow as tf
from pydantic import BaseModel, Field, PositiveInt
from typing import List, Optional, Dict, Tuple

class LstmModel:
    def __init__(self):
        self.model = None
    class SequentialConfig(BaseModel):
        units: List[PositiveInt] = Field(default=[64, ], description="滤波器数量，len控制lstm的层数")
        return_sequences: List[bool] = Field(default=[False, ],
                                             description="是否只在最后一个时间步产生输出，对应LSTM层数")
        output_shape: Tuple[PositiveInt, PositiveInt] = Field(default=(5, 19), description="调整reshape输出形状")

    def _validate_config(self, config: Optional[Dict], config_class: type) -> BaseModel:
        try:
            return config_class(**(config) or {})
        except Exception as e:
            raise ValueError(f"配置验证失败：{str(e)}")

    def _build_sequential_model(self,
                                config: dict = None
                                ) -> tf.keras.Sequential:

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
            optimizer=tf.optimizers.Adam(lr=0.001, epsilon=1e-07),
            loss=tf.losses.MeanSquaredError(),
            metrics=[tf.metrics.MeanAbsoluteError(name='mean_absolute_error')])

        return self

    def summary(self):
        """委托给内部的 Keras 模型"""
        if self.model is not None:
            return self.model.summary()
        else:
            print("模型尚未构建")

