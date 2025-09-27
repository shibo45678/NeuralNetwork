import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from typing import Union, List, Optional, Any
from keras.utils import timeseries_dataset_from_array
import matplotlib.pyplot as plt


class WindowGenerator:
    """数据窗口类与方法定义"""

    def __init__(self, input_width: int, label_width: int, shift: int,
                 label_columns: Optional[List[str]] = None):
        """
        shift: 从输入结束到标签开始之间的偏移（时间步数）Prediction starts shift hours after input
        label_columns:如果为None，则预测所有列
        """
        self.input_width = input_width  # input:6行（与split_window 窗口的inputs，input含义不相同，inputs说到列时，是包括XY特征）
        self.label_width = label_width  # label:5行
        self.shift = shift
        self.label_columns = label_columns

        # 行
        self.total_window_size = input_width + shift  # 标签的起始[索引]为shift
        self.label_start = self.total_window_size - self.label_width
        self.input_indices = np.arange(self.total_window_size)[0:input_width]  # [0:6]只是切片，不能直接赋值给对象，要在array上操作
        self.label_indices = np.arange(self.total_window_size)[self.label_start:]  # [2:]

        # 绘图用途
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}

    def __repr__(self):  # 返回对象的官方字符串表示
        return '\n'.join([
            f'Total window size : {self.total_window_size}',
            f'Input indices : {self.input_indices}',
            f'Label indices:{self.label_indices}',
            f'Label column name(s):{self.label_columns}'])

    """切分窗口数据"""

    @property
    def createTrainSet(self, data: Union[np.ndarray, pd.DataFrame]) -> tf.data.Dataset:
        return self.make_dataset(data)

    @property
    def createValSet(self,data: Union[np.ndarray, pd.DataFrame]) -> tf.data.Dataset:
        return self.make_dataset(self.val_df)

    @property
    def createTestSet(self,data: Union[np.ndarray, pd.DataFrame]) -> tf.data.Dataset:
        return self.make_dataset(self.test_df)


    @property
    def example(self) -> Any:
        """缓存模式，避免每次调用都从数据集中获取样本，每次相同用例
        """
        result = getattr(self, '_example', None)  # 尝试获取 _example 属性
        if result is None:
            result = next(iter(self.createTrainSet))  # 从训练集中获取第一个样本
            self._example = result  # 一对 inputs和labels
        return result
    # 列
    self.columns_indices = {name: i for i, name in enumerate(train_df.columns)}  # 包含X和Y的所有列



    def make_dataset(self, data: pd.DataFrame) -> tf.data.Dataset:
        """"""
        """
         将原始时间序列数据转换为TensorFlow 的Dataset对象。
         每个元素是一个(inputs, labels)对。
         包含批处理的时间序列窗口
         """
        data = np.array(data, dtype=np.float32)

        # 原始数据上操作，当设置batch_size=32时，数据集会将这些窗口组合成批次。
        # 每个batch(批/桶/组）有32个样本 ： 49025行 = 1532批 * 32样本/批
        # 每个样本是时间步length total_window_size的[sliding window]，每个元素是一个窗口（即一个样本）？
        ds = timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32
        )
        #  应用split_window函数将每个[批次]的数据分割为输入和标签
        ds = ds.map(self.split_window)

        return ds





    def split_window(self, features) -> tuple[tf.Tensor, tf.Tensor]:
        """"""
        """处理一个批次数据（包含 32 个样本/行），在32行上滑动窗口:
        features :三维张量[批大小, 时间步数, 特征数] 。特征数是包括"特征列(17) + 预测列(2:'T','p')
        labels: 标签张量，如果label_columns被指定，它只从labels中选择那些指定的列（否则就还是所有列）
        """

        inputs = features[:, 0:self.input_width, :]  # inputs是每个窗口的前input_width个时间步，所有特征(X+Y)17+2。
        labels = features[:, self.label_start:, :]  # 同理 X+Y

        # 匹配模型形状
        if self.label_columns is not None:

            labels = tf.stack(
                # 选择所有批次、所有时间步、但只有特定列的数据（在原labels张量，直接用索引进行特征列筛选： 19->2）
                # 子张量，每列的形状为[batch_size, label_width]（即去掉了特征维度）
                # tf.stack 对于每个时间步，T和p的值被组合在一起(从分开的列压缩到一个维度)，形成一个新的特征维度。不是melt效果数据变长.
                [labels[:, :, self.columns_indices[name]] for name in self.label_columns],
                axis=-1
            )

            print(f" stack后的labels 形状: {labels.shape}")
            # 确保批处理大小和时间步数正确，但特征数允许变化（用None表示）
            inputs.set_shape([None, self.input_width, None])
            labels.set_shape([None, self.label_width, None])

            return inputs, labels



    def window_plot(self, model=None, plot_col='T', max_subplots=3):
        inputs, labels = self.example # 使用TensorFlow的迭代机制，才能正确解包inputs和label
        plot_col_index = self.columns_indices[plot_col]
        max_n = min(max_subplots, len(inputs))

        for n in range(max_n):
            plt.subplot(3, 1, n + 1)
            plt.ylabel(f'标准化后的 {plot_col}')

            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='green', s=64)

            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='orange', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
        plt.show()