import tensorflow as tf
from pydantic import BaseModel, Field, validator, PositiveInt
from pydantic.dataclasses import dataclass
from typing import List,Optional,Dict


class LstmModel:

    class SequentislConfig(BaseModel):
        filters:List[PositiveInt] = Field(default=[64,],description="滤波器数量，len控制lstm的层数")
        return_sequences:bool = Field(default=False,description="是否只在最后一个时间步产生输出")
        dense_units: List[PositiveInt] = Field(default=[5 * 19],description="全连接层单元数。值根据输出label形状定，如果只预测5行2列特征5*2；len控制全连接层层数。")
        output_shape: List[PositiveInt,PositiveInt]=Field(default =[5,19],description="调整reshape输出形状")
    @validator('output_shape')
    def validate_output_shape_consistency(cls,v,values):
        if 'dense_units' in values and v[1]*v[0] !=values['dense_units']:
            raise ValueError(
                f"输出的形状长{v[0]} 和宽{v[1]}相乘的结果，必须与{values['dense_units']} 保持一致"
            )
        return v

    def _validate_config(self,config:Optional[Dict],config_class:type)-> BaseModel:
        try:
            return config_class(**(config) or {})
        except Exception as e:
            raise ValueError(f"配置验证失败：{str(e)}")

    def __init__(self):
        pass

    def _build_sequential_model(self,
                                config: dict = None
                                ) -> tf.keras.Sequential:

        """参数检查"""
        model_config = self._validata_config(config, self.SequentialConfig)

        filters = model_config.filters
        return_sequences = model_config.return_sequences
        dense_units = model_config.dense_units
        output_shape =model_config.output_shape

        model = tf.keras.Sequential()

        # 添加LSTM层
        for i ,(f,s) in enumerate(filters,return_sequences):
            model.add(tf.keras.layers.LSTM(filters=f,activation='tanh',return_sequences=s) # 确认return
                      )
            # shape[32,6,19]==>[32,64]
            # tanh 将一个实数映射到（-1 1）的区间
            print(f"添加LSTM层lstm_{i + 1}:Filter={f},Activation='tanh',Return_sequences={s}")

        # 添加全连接层
        model.add(tf.keras.layers.Dense(units=dense_units,kernel_initializer=tf.initializers.zeros)) # dense  shape[32,5*19]
        print(f"添加Dense层:Units={dense_units},设置全零初始化kernel_initializer")

        # 输出层,调整形状
        model.add(tf.keras.layers.Reshape(output_shape)) # [32,5,19]

        # 模型编译
        model.compile(
            optimizer=tf.optimizers.Adam(lr=0.001,epsilon=1e-07),
            loss=tf.losses.MeanSquaredError(),
            metrics=[tf.metrics.MeanAbsoluteError()])

        return model

