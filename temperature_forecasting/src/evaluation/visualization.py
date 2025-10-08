import matplotlib.pyplot as plt
import numpy as np

# def history_plot(train_metric:np.ndarray,val_metric:np.ndarray,
#                  title:str,label:str):
#     plt.figure(figsize=(9, 7))  # 设置图形宽度和高度
#
#     epochs = np.arange(1, len(train_metric) + 1)  # 设置横坐标epochs的计算方法
#     plt.clf()  # 清除所有轴，保持窗口打开
#
#     plt.plot(epochs, train_metric, 'r', label="Training")
#     plt.plot(epochs, val_metric, 'b', label="Validation")
#     plt.title(title)
#     plt.xlabel('Epochs')
#     plt.ylabel(label)
#     plt.legend()
#     plt.show()


def history_plot(history, model_name=""):
    """通用训练历史绘图函数"""

    # 获取所有可用的指标
    available_metrics = list(history.history.keys())
    print(f"可用的指标: {available_metrics}")

    # 分离损失和其他指标
    loss_metrics = [m for m in available_metrics if 'loss' in m and not m.startswith('val_')]
    other_metrics = [m for m in available_metrics if 'loss' not in m and not m.startswith('val_')]

    # 创建子图
    n_plots = 1 + len(other_metrics)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))

    if n_plots == 1:
        axes = [axes]

    # 绘制损失
    epochs = np.arange(1, len(history.history[loss_metrics[0]]) + 1)

    axes[0].plot(epochs, history.history[loss_metrics[0]], 'r-', label=f'Training {loss_metrics[0]}')
    if f'val_{loss_metrics[0]}' in history.history:
        axes[0].plot(epochs, history.history[f'val_{loss_metrics[0]}'], 'b-', label=f'Validation {loss_metrics[0]}')
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # 绘制其他指标
    for i, metric in enumerate(other_metrics, 1):
        axes[i].plot(epochs, history.history[metric], 'g-', label=f'Training {metric}')
        if f'val_{metric}' in history.history:
            axes[i].plot(epochs, history.history[f'val_{metric}'], 'orange', label=f'Validation {metric}')
        axes[i].set_title(f'{model_name} - {metric}')
        axes[i].set_xlabel('Epochs')
        axes[i].set_ylabel(metric)
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()


