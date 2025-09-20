from ..utils import WindowGenerator
from ..models import CnnModel,LstmModel
import matplotlib.pyplot as plt
from typing import Union


def evaluate_model(name:str,
                   model:Union['CnnModel','LstmModel'],
                   window:'WindowGenerator'):

    window.window_plot(model)
    plt.show()

    """MAE"""
    # 用验证集、测试集评估模型，并返回验证集评估结果（损失值和MAE）evaluate
    val_performance = model.evaluate(window.createValSet,verbose=0)
    test_performance = model.evaluate(window.createTestSet, verbose=0)

    # 找出测试值MAE所属的索引(指标为损失值-均方误差和MAE)
    metric_index = model.metrics_names.index('mean_absolute_error')
    print(metric_index)
    # 根据MAE的索引遍历验证集的评估结果，返回所有模型的MAE测量值
    val_mae =val_performance[metric_index]
    print(f"{name}模型的验证集平均绝对值误差：{val_mae}")
    test_mae = test_performance[metric_index]
    print(f"{name}模型的测试集平均绝对值误差：{test_mae}")
    print(test_mae)


    return val_mae,test_mae

