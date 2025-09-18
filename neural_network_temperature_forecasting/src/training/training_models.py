from ..utils import WindowGenerator
import tensorflow as tf

MAX_EPOCHS=20 # 训练总轮数
def TrainingCnn(model,window,epochs=MAX_EPOCHS,verbose=2):

        history = model.fit(
            window.createTrainSet,
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
                    filepath='best_model.h5',  # 保存路径
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
                ),

                # # TensorBoard日志
                # tf.keras.callbacks.TensorBoard(
                #     log_dir='./logs',
                #     histogram_freq=1
                # )


                ]
        )

        # 加载最佳模型进行预测或继续训练
        best_model = tf.keras.models.load_model('best_model.h5')
        # predictions = best_model.predict(single_window.createValSet)

        return history,best_model



def TrainingLstm(model,window):
    history = fit(model, window)
    # LSTM 层的参数总数【（64+19）*64 + 64】*4 == 【（上一轮输出+输入）*（全联接输出）+（输出层偏置）】*4层（遗忘门*1+记忆门*2+输出门*1）
    # dense 1  参数总数 64*95+95=6175 全联接
    multi_lstm_model.summary()
