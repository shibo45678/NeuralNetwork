from ..utils import WindowGenerator
import tensorflow as tf

MAX_EPOCHS=20 # 训练总轮数
def TrainingCnn(model:tf.keras.models,
                window:'WindowGenerator',
                epochs:int=MAX_EPOCHS,
                verbose:int=2,
                file_path:str='best_model_cnn.h5'):

        history = model.fit(
            window.createTrainSet, # x,y
            validation_data=window.createValSet,
            epochs=epochs,
            verbose=verbose, # 设置日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录 2 epoch每轮输出一行记录
            callbacks=[
                # 早停：防止过拟合
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', # 监测指标
                                                 patience=5, # 没有进步的训练轮数，在这之后训练停止
                                                 mode='min', # 当监测指标停止减少时训练停止（维持最小值）
                                                 restore_best_weights=True),

                # 模型检查点：保存最佳模型
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=file_path,  # 保存路径
                    monitor='val_loss',  # 监控指标
                    save_best_only=True,  # 只保存最佳模型
                    save_weights_only=False,  # 保存整个模型（包括结构）
                    verbose=1  # 显示保存信息
                ),

                # 添加学习率调度 提升训练效果
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,  # 学习率减半
                    patience=2,  # 2个epoch无改善就降低LR
                    min_lr=1e-6,  # 最小学习率
                    verbose=2
                )
                ]
        )

        # 加载最佳模型进行预测或继续训练
        best_model_cnn = tf.keras.models.load_model(file_path)
        # predictions = best_model.predict(single_window.createValSet)

        return history,best_model_cnn



def TrainingLstm(model:tf.keras.models,
                 window:'WindowGenerator',
                 epochs:int=100,
                 verbose:int=2,
                 file_path:str ='best_model_lstm.h5'):

    history = model.fit(window.createTrainSet, # 或者直接写window 训练数据也可以不写
                        validation_data=window.createValSet,
                        epochs=epochs,
                        verbose=verbose,
                        callbacks=[
                            # 早停：防止过拟合
                            tf.keras.callbacks.EarlyStopping(monitor='val_loss',  # 监测指标
                                                             patience=5,  # 没有进步的训练轮数，在这之后训练停止
                                                             mode='min',  # 当监测指标停止减少时训练停止（维持最小值）
                                                             restore_best_weights=True),

                            # 模型检查点：保存最佳模型
                            tf.keras.callbacks.ModelCheckpoint(
                                filepath=file_path,  # 保存路径
                                monitor='val_loss',  # 监控指标
                                save_best_only=True,  # 只保存最佳模型
                                save_weights_only=False,  # 保存整个模型（包括结构）
                                verbose=1  # 显示保存信息
                            ),

                            # 添加学习率调度 提升训练效果
                            tf.keras.callbacks.ReduceLROnPlateau(
                                monitor='val_loss',
                                factor=0.5,  # 学习率减半
                                patience=2,  # 2个epoch无改善就降低LR
                                min_lr=1e-6,  # 最小学习率
                                verbose=2
                            )
                        ]
                        )
    # 加载最佳模型进行预测或继续训练
    best_model_lstm = tf.keras.models.load_model(file_path)

    return history,best_model_lstm

