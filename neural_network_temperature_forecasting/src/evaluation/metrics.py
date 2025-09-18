from ..utils import WindowGenerator
from ..models import CnnModel
import matplotlib.pyplot as plt



def EvaluateModel(name:str,model:'CnnModel',window:'WindowGenerator'):
    val_performance = {}
    test_performance = {}

    # 用验证集、测试集评估模型，并返回验证集评估结果（损失值和MAE）evaluate
    val_performance[name] = model.evaluate(window.createValSet,verbose=0)
    test_performance[name] = model.evaluate(window.createTestSet, verbose=0)
    # 输出评估结果到multi_performance和multi_val_performance字典里
    # 字典到key对应不同模型名称，value对应不同模型下的训练结果（指标为损失值-均方误差和MAE）

    window.window_plot(model)
    plt.show()

