""" CNN-LSTM 特殊处理"""
from ..data.processing import DataPreprocessor
from ..data.exploration import Visualization
import pandas as pd
import numpy as np
import datetime


class SpecialCnnLstm():
    def __init__(self, df: pd.DataFrame):
       self.df =df

    def prepare_for_cnn_lstm(self) ->'SpecialCnnLstm' :
        """1.处理'Data Time' 时间列
           2.处理风向'wd'与风速'wv'列数据"""

        """1. 处理'Data Time'"""
        # a.将'Data Time'列数据从str数据类型转换为datatime数据类型，保存到新变量data_time中（原df中删除'Data Time'列）
        date_time = pd.to_datetime(self.df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
        print(date_time)

        # b.将data_time中数据转换为时间戳格式的数据
        datetime.datetime.timestamp(date_time[5])
        timestamp_s = date_time.map(datetime.datetime.timestamp)
        print(timestamp_s)

        # c.将时刻序列映射为正弦曲线序列
        day = 24 * 60 * 60  # 一天多少秒
        year = (365.2425) * day  # 一年多少秒

        self.df['Day sin'] = np.sin((timestamp_s * 2 * np.pi) / day)
        self.df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        self.df['Year sin'] = np.sin((timestamp_s / year) * 2 * np.pi)
        self.df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
        print("新增4列：['Day sin', 'Day cos', 'Year sin', 'Year cos']")
        print(self.df[['Day sin', 'Day cos', 'Year sin', 'Year cos']])

        # d.将转换结果可视化
        viz = Visualization()
        viz.plot_time_signals(X=np.array(self.df['Day sin'])[:25],  # 24小时
                 y=np.array(self.df['Day cos'])[:25],
                 xlabel='时间[单位：时]（Time [h]）',
                 title='一天中的时间信号（Time of day signal）')

        """2.处理风向'wd'与风速'wv'列数据"""

        # 目标效果：处理前用极坐标（风速m/s）和风向（0-360）来描述风的强度和方向，
        # 处理后:用正交坐标系的两个维度（x轴和y轴）上风的强度来描述上述风的强度和方向 ['Wx', 'Wy', 'max Wx', 'max Wy']
        print(self.df[['wv', 'max. wv', 'wd']])  # 平均风速、最大风速、风向（角度制）

        # 处理步骤：
        # a.将风向和风速列数据转换为风矢量，重新存入原数据框中
        # b.2D直方图--通过可视化的方式解释风矢量类型的数据由于原表风速和风向数据的原因

        # 原表风速和风向数据
        Visualization.plot_hist2d(x=self.df['wd'],
                                  y=self.df['wv'],
                                  xlabel='风向 [单位：度]',
                                  ylabel='风速 [单位：米/秒]')

        # 风矢量类型的数据
        wv = self.df.pop('wv')  # 先抓出 再丢了 将df中的wv列保存到wv中，并从原来的df中删除
        max_wv = self.df.pop('max. wv')
        wd_rad = self.df.pop('wd') * np.pi / 180  # 风向由角度制转换为弧度制

        self.df['Wx'] = wv * np.cos(wd_rad)  # 计算平均风力wv的x和y分量，保存到df的'Wx'列和'Wy'列中
        self.df['Wy'] = wv * np.sin(wd_rad)

        self.df['max Wx'] = max_wv * np.cos(wd_rad)  # 计算最大风力'max. mv'的x和y分量，保存到df的'max Wx'列和'max Wy'列中
        self.df['max Wy'] = max_wv * np.sin(wd_rad)

        # 不需要初始化任何东西，最适合静态方法，然后类名调用
        Visualization.plot_hist2d(x=self.df['Wx'],
                                  y=self.df['Wy'],
                                  xlabel='风的X分量[单位：m/s]',
                                  ylabel='风的Y分量[单位：m/s]')

        Visualization.plot_hist2d(x=self.df['max Wx'],
                                  y=self.df['max Wy'],
                                  xlabel='最大风的X分量[单位：m/s]',
                                  ylabel='最大风的Y分量[单位：m/s]')

        # 对比两图，分解后有利于我们观察风的状况：找到原点（0，0），
        # 假设向上为北，那么南方向的 风出现次数较多，此外我们还可以观察到东北-西南方向的风

        return self

    def get_special_processed_data(self):
        return self.df
