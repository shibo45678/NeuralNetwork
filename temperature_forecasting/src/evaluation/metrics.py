from ..utils import WindowGenerator
from ..models import CnnModel, LstmModel
import matplotlib.pyplot as plt
from typing import Union


# model.metrics_names 包含的是评估时的指标名称
# history.history 包含的是训练时的指标名称（包括验证集指标）
def evaluate_model(name: str,
                   model: Union['CnnModel', 'LstmModel'],
                   window: 'WindowGenerator',
                   valsets,
                   testsets,
                   ):

    window.window_plot(model)
    plt.show()

    print(f"可用指标 metrics_names:{model.metrics_names}") # ['loss', 'compile_metrics']

    """MAE"""
    # 用验证集、测试集评估模型，并返回验证集评估结果（损失值和MAE）evaluate
    val_performance = model.evaluate(valsets, verbose=0)
    test_performance = model.evaluate(testsets, verbose=0)

    print(f"验证集评估结果：{dict(zip(model.metrics_names,val_performance))}")
    print(f"测试集评估结果：{dict(zip(model.metrics_names,test_performance))}")

    # 找出测试值MAE所属的索引(指标为损失值-均方误差和MAE)
    metric_index = model.metrics_names.index('compile_metrics') # 编译的名字
    print(metric_index)
    # 根据MAE的索引遍历验证集的评估结果，返回所有模型的MAE测量值
    val_mae = val_performance[metric_index]
    print(f"{name}模型的验证集平均绝对值误差：{val_mae}")
    test_mae = test_performance[metric_index]
    print(f"{name}模型的测试集平均绝对值误差：{test_mae}")
    print(test_mae)

    return val_mae, test_mae
