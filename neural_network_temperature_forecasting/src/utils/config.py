import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 在 import 之前设置！
import tensorflow as tf


class TensorFlowConfig:
    """TensorFlow配置类"""

    @staticmethod
    def setup_environment():
        """设置TensorFlow运行环境"""
        # 隐藏信息性日志
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # 配置GPU内存增长（如果有GPU）
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

    @staticmethod
    def check_performance():
        print("TensorFlow版本:", tf.__version__)
        print("可用GPU数量:", len(tf.config.experimental.list_physical_devices('GPU')))






