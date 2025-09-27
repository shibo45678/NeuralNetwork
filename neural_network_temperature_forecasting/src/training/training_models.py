from ..evaluation import history_plot
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


def TrainingModel(model_name:str,
                model,  # tf.keras.models
                window,  # 'WindowGenerator'
                epochs: int = 20, # 总轮数
                verbose: int = 2):

    weights_path = f"best_model_{model_name}_weights.h5"

    record = model.fit(
        window.createTrainSet,  # x,y
        validation_data=window.createValSet,
        epochs=epochs,
        verbose=verbose,  # 设置日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录 2 epoch每轮输出一行记录
        callbacks=[
            # 早停：防止过拟合
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',  # 监测指标
                                             patience=8,  # 没有进步的训练轮数，在这之后训练停止
                                             mode='min',  # 当监测指标停止减少时训练停止（维持最小值）
                                             min_delta=0.00001,  # 设置最小改善阈值
                                             restore_best_weights=True),

            # 模型检查点：保存最佳模型
            tf.keras.callbacks.ModelCheckpoint(
                filepath=weights_path,  # 保存路径
                monitor='val_loss',  # 监控指标
                save_best_only=True,  # 只保存最佳模型
                save_weights_only=True,  # False 保存整个模型（包括结构）,True 保存参数
                verbose=1,  # 显示保存信息
                save_freq='epoch'  # 默认就是每个epoch保存，可以改为按批次保存
            ),

            # 添加学习率调度 提升训练效果
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # 学习率减半
                patience=3,  # 2个epoch无改善就降低LR
                min_lr=1e-7,  # 最小学习率
                verbose=2
            )
        ]
    )

    """训练过程可视化"""
    # record.history 为字典对象，包含训练过程中的loss的测量指标等记录项
    history_plot(history=record, model_name=model_name)

    return record, weights_path




# 一般训练规律 损失值：
# train loss 不断下降   validation loss不断下降---网络仍在学习
# train loss 不断下降   validation loss不断上升---网络过拟合，添加dropout和max pooling
# train loss 不断下降   validation loss趋于不变---网络欠拟合
# train loss 趋于不变   validation loss趋于不变---网络陷入瓶颈，减小学习率（自适应效果不大）和batch数量减少
# train loss 不断上升   validation loss不断上升---网络结构问题，训练超参数设置不当，数据集需要清洗等
# train loss 不断上升   validation loss不断下降---数据集有问题，建议重新选择


# 一般训练规律：准确度（整体训练趋势）
# train accuracy 不断上升   validation accuracy 不断上升---网络仍在学习
# train accuracy 不断上升   validation accuracy 不断下降---网络过拟合，添加dropout和max pooling
# train accuracy 不断上升   validation accuracy 趋于不变---网络欠拟合
# train accuracy 趋于不变   validation accuracy 趋于不变---网络陷入瓶颈，减小学习率（自适应效果不大）和batch数量减少
# train accuracy 不断下降   validation loss 不断下降---网络结构问题，训练超参数设置不当，数据集需要清洗等
# train accuracy 不断下降   validation loss 不断上升---数据集有问题，建议重新选择


