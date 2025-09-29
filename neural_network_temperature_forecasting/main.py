from src.data.processing import DataPreprocessor  # 类使用绝对路径
from src.data.exploration import Visualization
from src.utils.windows import WindowGenerator
from src.models.cnn import CnnModel
from src.models.lstm import LstmModel
from src.training.training_models import TrainingModel
from src.evaluation.metrics import evaluate_model
from .trained.trained import ReconstructPredictor
from src.utils.config import TensorFlowConfig
import time
import matplotlib.pyplot as plt
from src.utils.debug_controller import DebugController
import os
import sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def main():
    TensorFlowConfig.setup_environment()
    # 初始化调试控制器，指定监控的文件
    debug_ctrl = DebugController()
    current_file = __file__  # 当前文件路径

    # 集中预声明所有可能的变量,避免黄色警告
    # train_inputs = None
    # train_labels = None
    # cnn_model = None
    # single_window = None
    # val_mae_cnn = None
    # val_mae_lstm1 = None
    # val_mae_lstm2 = None
    # test_mae_cnn = None
    # test_mae_lstm1 = None
    # test_mae_lstm2 = None
    # timeseries_cnn_model = None

    print(f"当前会话状态:")
    print(debug_ctrl.get_session_info())

    if debug_ctrl.continue_from_breakpoint('after_data_preparation', locals(), __file__):
        print("从 after_data_preparation 阶段继续")
    else:
        print("=== 重新执行 data_preparation 阶段 ===")
        print("包括：清洗数据 + 划分数据集 + 构建窗口数据...")
        # STAGE_START:after_data_preparation
        config= [
            ('minmax', {'feature_range': (0, 1), 'columns': ['T']}),
            # ('std_scaler', {'columns': []}),
            # ('robust_scaler', {'quantile_range': (25, 75), 'columns': []})
        ]
        preprocessor = (DataPreprocessor(input_files=["data_climate.csv"])
        .handleEncoding()  # 编码utf-8
        .load_all_data(pattern="new_*.csv")
        .describe_data()
        .problem_columns_fixed(problem_columns=[])
        .special_columns_fixed(problem_columns=['T'])
        .identify_column_types()
        .process_numeric_data()
        .encode_categorical_data()
        .process_other_data()
        .handle_missing_values(cat_strategy='mode', num_strategy='median')
        .remove_duplicates()
        .delete_useless_cols(target_cols=None)
        .check_extreme_features({'name': 'iqr', 'threshold': 1.5})  # 查看
        .check_extreme_features({'name': 'zscore', 'threshold': 3})
        .check_extreme_features({'name': 'multivariate', 'contamination': 0.025})  # 预期异常比例
        .systematic_resample(start_index=5, step=6)  # 切片，从第一小时开始（索引5开始），每隔6个(6*10分钟)采一次
        .remove_outliers(method='custom')  # 目前仅处理了少数物理异常
        .handle_time_col(col='Date Time', format='%d.%m.%Y %H:%M:%S')  # time_col正余弦
        .handle_vec_col(dir_cols=['wd'], var_cols=['wv', 'max. wv'])  # vec_col 风矢量 要求顺序
        .train_val_test_split(train_size=0.7, val_size=0.2, test_size=0.1)
        .unify_feature_scaling(transformers=config))  # 独热编码 / 分类型 / 时间不处理

        # 保存预处理结束时的scalar配置
        scalers_path = preprocessor.save_scalers(filename='scalers.pkl')
        constant_path = preprocessor.save_constant_values(filename='constant.pkl')

        print("\n处理历史:")
        history = preprocessor.get_histroy()
        print(history)

        # 画小提琴图（观察整个df标准化后的数据分布)
        df_std = preprocessor.get_data()
        df_std = df_std.melt(var_name='Column', value_name='normalized')  # 宽表变长表，数据形状匹配

        viz = Visualization()
        viz.violin_plot(df=df_std,
                        var_name='Column', value_name='normalized',  # Standardized
                        title="统计分布小提琴图")

        """构建窗口数据"""
        # 1 使用WindowGenerator类实例 构造窗口数据
        df_train = preprocessor.get_train_val_test_data()[0]
        df_val = preprocessor.get_train_val_test_data()[1]
        df_test = preprocessor.get_train_val_test_data()[2]

        # 指定预测特征列
        single_window = WindowGenerator(input_width=6, label_width=5, shift=24, label_columns=['T', 'p'])
        print(single_window)

        # 2 构建训练集、验证集和测试集
        print('训练数据：')
        window_train_data = single_window.createDataset(df_train)
        print(window_train_data)
        # 正确获取inputs和labels的形状
        # createTrainSet()返回的是tf.data.Dataset对象，不是普通的元组列表
        # 必须使用TensorFlow的迭代机制：iter(), take()正确解包inputs和labels ：for... .take(1) / example有iter
        train_inputs, train_labels = single_window.example
        print(f"train_inputs 形状: {train_inputs.shape}")
        print(f"train_labels 形状: {train_labels.shape}")

        print('验证数据：')
        window_val_data = single_window.createDataset(df_val)
        print(window_val_data)
        print('测试数据：')
        window_test_data = single_window.createDataset(df_test)
        print(window_test_data)

        # 画个训练集的图
        single_window.window_plot(plot_col='T')

        # STAGE_END:after_data_preparation
        debug_ctrl.save_debug_session(locals(), 'after_data_preparation', __file__)

    """=========================================CNN========================================="""
    if debug_ctrl.continue_from_breakpoint('after_model_build_cnn', locals(), __file__):
        print("从 after_model_build_cnn 阶段继续")
    else:
        print("==== 重新执行 model_build_cnn 阶段 ====")
        print("包括：构建、编译...")
        # STAGE_START:after_model_build_cnn

        # 基于历史6个时间点的天气情况（6行19列）预测经过24小时（shift=24)未来5个时间点 'T''p'列
        timeseries_cnn_model = CnnModel(architecture_type='parallel')  # 分支并行模式
        config_cnn_parallel_model = {
            'input_shape': train_inputs.shape[1:],  # 去掉batch的形状
            'output_shape': train_labels.shape[1:],
            'branch_filters': [[32, 32], [64, 64]],
            'branch_kernels': [[2, 3], [2, 3]],
            'branch_dilation_rate': [[1, 1], [1, 1]],
            'activation': 'relu'}  # 或者'swish'
        cnn_model = timeseries_cnn_model._build_parallel_model(config_cnn_parallel_model)

        # STAGE_END:after_model_build_cnn
        debug_ctrl.save_debug_session(locals(), 'after_model_build_cnn', __file__)

    if debug_ctrl.continue_from_breakpoint('after_training_cnn', locals(), __file__):
        print("从 after_training_cnn 阶段继续")
    else:
        print("====重新执行 training_cnn 阶段 ====")
        print("包括：CNN 模型的训练 + 评估")
        # STAGE_START:after_training_cnn

        # 训练模型
        history_cnn, cnn_weights_path = TrainingModel(model_name='cnn', model=cnn_model,
                                                      trainset=window_train_data,
                                                      valset=window_val_data,
                                                      verbose=2, epochs=20)
        timeseries_cnn_model.summary()  # 出来一个表 显示每一层参数个数

        # 重构模型
        reconstr_cnn = (ReconstructPredictor()
                        .reconstruct_trained_model(original_model=cnn_model, weights_path=cnn_weights_path)
                        .get_constr_model())

        # 评估模型
        val_mae_cnn, test_mae_cnn = evaluate_model(name='cnn_model', model=reconstr_cnn,
                                                   window=single_window,
                                                   valset=window_val_data,
                                                   testset=window_test_data)
        print(f"评估模型 best_model_cnn 的验证集和测试集的均方绝对值误差MAE结果如下：")
        print(val_mae_cnn, test_mae_cnn)

        # STAGE_END:after_training_cnn
        debug_ctrl.save_debug_session(locals(), 'after_training_cnn', __file__)

    """=========================================LSTM1========================================="""
    if debug_ctrl.continue_from_breakpoint('after_model_lstm1', locals(), __file__):
        print("==== 从after_model_lstm1 阶段继续 ====")
    else:
        print("==== 重新执行 model_lstm1 阶段 ====")
        print("包括：lstm1模型的构建、编译、训练、评估...")
        # STAGE_START:after_model_lstm1

        timeseries_lstm1_model = LstmModel()
        config_lstm1 = {
            'units': [64, ],
            'return_sequences': [False, ],  # 只输出最后一行
            'output_shape': train_labels.shape[1:]}

        lstm1_model = timeseries_lstm1_model._build_sequential_model(config_lstm1)
        """训练LSTM模型1"""
        history_lstm1, lstm1_weights_path = TrainingModel(model_name='lstm1', model=lstm1_model,
                                                          trainset=window_train_data,
                                                          valset=window_val_data, verbose=2, epochs=100)
        timeseries_lstm1_model.summary()  # 参数个数

        # 重构模型
        reconstr_lstm1 = (ReconstructPredictor()
                          .reconstruct_trained_model(original_model=lstm1_model, weights_path=lstm1_weights_path)
                          .get_constr_model())

        """评估LSTM模型1"""
        val_mae_lstm1, test_mae_lstm1 = evaluate_model(name='lstm1', model=reconstr_lstm1,
                                                       window=single_window,
                                                       valset=window_val_data, testset=window_test_data)
        print(f"评估模型 best_model_lstm1 的验证集和测试集的均方绝对值误差MAE结果如下：")
        print(val_mae_lstm1, test_mae_lstm1)

        # STAGE_END:after_model_lstm1
        debug_ctrl.save_debug_session(locals(), 'after_model_lstm1', __file__)

    """=========================================LSTM2========================================="""
    if debug_ctrl.continue_from_breakpoint('after_model_lstm2', locals(), __file__):
        print("==== 从 after_model_lstm2 阶段继续 ====")
    else:
        print("==== 重新执行 model_lstm2 阶段 ====")
        print("包括：lstm2模型的构建、编译、训练、评估")
        # STAGE_START:after_model_lstm2

        timeseries_lstm2_model = LstmModel()
        config_lstm2 = {
            'units': [64, 64],  # 2层LSTM
            'return_sequences': [True, False],  # 只输出最后一行
            'output_shape': train_labels.shape[1:]}
        lstm2_model = timeseries_lstm2_model._build_sequential_model(config_lstm2)

        """训练LSTM模型2"""
        history_lstm2, lstm2_weights_path = TrainingModel(model_name='lstm2', model=lstm2_model,
                                                          trainset=window_train_data,
                                                          valset=window_val_data, verbose=2, epochs=100)
        timeseries_lstm2_model.summary()  # 参数个数

        # 重构模型
        reconstr_lstm2 = (ReconstructPredictor()
                          .reconstruct_trained_model(original_model=lstm2_model, weights_path=lstm1_weights_path)
                          .get_constr_model())

        """评估LSTM模型2"""
        val_mae_lstm2, test_mae_lstm2 = evaluate_model(name='lstm2', model=reconstr_lstm2, window=single_window,
                                                       trainset=window_train_data, valset=window_val_data)
        print(f"评估模型 best_model_lstm2 的验证集和测试集的均方绝对值误差MAE结果如下：")
        print(val_mae_lstm2, test_mae_lstm2)

        # STAGE_END:after_model_lstm2
        debug_ctrl.save_debug_session(locals(), 'after_model_lstm2', __file__)

    """=========================================比较CNN和LSTM的预测效果========================================="""
    if debug_ctrl.continue_from_breakpoint('after_models_compare', locals(), __file__):
        print("==== 从 after_models_compare 阶段继续 ====")
    else:
        print("==== 重新执行models_compare ====")
        print("包括：cnn,lstm1,lstm2 模型的 MAE对比")
        # STAGE_START:after_models_compare

        # 画出每个模型里面测试集和验证集的MAE
        val_mae = [val_mae_cnn, val_mae_lstm1, val_mae_lstm2]
        test_mae = [test_mae_cnn, test_mae_lstm1, test_mae_lstm2]
        x = len(val_mae)  # 3个模型

        plt.ylabel('mean_absolute_error')  # 指定纵轴标签
        plt.bar(x=x - 0.17, height=val_mae, width=0.3, label='Validation')
        plt.bar(x=x + 0.17, height=test_mae, width=0.3, label='Test')
        plt.xticks(ticks=x, labels=['conv1D', 'lstm1', 'lstm2'], rotation=45)
        _ = plt.legend()

        # STAGE_END:after_models_compare
        debug_ctrl.save_debug_session(locals(), 'after_models_compare', __file__)

    """=========================================预测========================================="""


if __name__ == "__main__":
    main()
