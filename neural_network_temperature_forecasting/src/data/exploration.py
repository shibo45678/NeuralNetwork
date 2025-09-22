import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
import plotly.graph_objects as go

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class Visualization:

    def plot_time_signals(self, X: np.ndarray = None, y: np.ndarray = None, xlabel: str = '', title: str = ''):
        """线图"""
        plt.figure(figsize=(12, 6))
        plt.plot(X, label='X数据', color='blue')  # 24小时
        plt.plot(y, label='y数据', color='red')
        plt.xlabel(xlabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_hist2d(x: pd.Series , y: pd.Series , xlabel: str , ylabel: str ):
        plt.hist2d(x=x, y=y, bins=(50, 50), vmax=400)
        plt.colorbar()
        plt.xlabel(xlabel=xlabel)
        plt.ylabel(ylabel=ylabel)
        plt.show()

    # @staticmethod
    # def plot_violinplot(x:str='Column',y:str='Standardized',data:pd.DataFrame=pd.DataFrame()):
    #     plt.figure(figsize=(12, 6))
    #     ax = sns.violinplot(x=x, y=y, data=data)
    #     _ = ax.set_xticklabels(data.keys(), rotation=90)
    #     plt.show()
        # 小提琴图：每个小提琴展示了原数据中每一列数据的统计特征，例如第二个小提琴表示列温度列数据可能出现的取值，以及这些取值出现的概率。
        #  上下端点纵坐标值是可能的取值，每个取值在横坐标上的宽度表示该取值出现的概率。
        #  例如第二个小提琴中，越宽的地方表示温度出现概率越高。
        #  每个小提琴里的矩形上下端点表示四分之一和四分之三位数的位置，白点表示二分位数位置

    def violin_plot(self,df:pd.DataFrame,
                    var_name: str , value_name: str,
                    title:str): # fences 是用来识别异常值（outliers） 的边界

        fig = go.Figure()
        for column in df[var_name].unique():
            fig.add_trace(go.Violin(
                x=df[df[var_name] == column][var_name],
                y=df[df[var_name] == column][value_name],
                name=column,

                # 均线设置（必须用字典）
                meanline={
                    'visible': True,  # 显示均线
                    'color': 'red',  # 均线颜色
                    'width': 2  # 均线宽度
                },
                # 箱线图设置（也必须用字典）
                box={
                    'visible': True,  # 显示箱线图
                    'fillcolor': 'blue',  # 箱线图颜色
                    'width': 0.2  # 箱线图宽度
                },

                # points='all', #  显示数据点：'all', 'outliers', 'suspectedoutliers', False
                # pointpos=-1.5,  # 数据点位置
                # jitter=0.1,  # 数据点抖动
                # bandwidth=0.5,  # 核密度估计带宽
                # side='both',  # 'both', 'positive', 'negative'
                #
                # showlegend=True
            ))

        # 添加图表布局和标题
        fig.update_layout(
            title=title,
            xaxis_title=var_name,
            yaxis_title=value_name,
            width=800,
            height=500
        )

        fig.show(renderer="browser")


















