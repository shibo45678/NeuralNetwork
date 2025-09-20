from src.data import EncodingHander, DataLoader, FixProblemColumns
from src.data.processing import DataPreprocessor, DataResampler, DataSplitter  # 类使用绝对路径
from src.data.extreme_processing import ExtremeDataHandler
from src.data.special_processing_cnn_lstm import SpecialCnnLstm
from src.data.exploration import Visualization
from src.utils.windows import WindowGenerator
from src.models import CnnModel
from src.models import LstmModel
from src.training import TrainingCnn,TrainingLstm
from src.evaluation import evaluate_model
import pandas as pd
import time
import matplotlib.pyplot as plt


def main():
    """"""
    """编码utf-8"""
    file_names = ["data_climate.csv"]
    encoder = EncodingHander(file_names)
    encoder.handleEncoding()

    print("文件encoding中，请等待...")
    time.sleep(10)  # 等待10秒
    print("数据加载完成！")

    """加载"""
    loader = DataLoader()
    data = loader.load_all_data(pattern="new_*.csv")
    print(data.head(5))

    """修复"""
    fix = FixProblemColumns(loader, problem_columns=['T'])
    fixed_data = fix.special_columns_fixed().get_fixed_data()

    """清洗"""
    # 使用预处理类，先执行一般处理流程
    preprocessor = DataPreprocessor(fixed_data)

    (preprocessor
     .identify_column_types()
     .process_numeric_data()
     .encode_categorical_data()
     .handle_missing_values(cat_strategy='custom', num_strategy='mean')
     .remove_duplicates()
     .create_extreme_features_zscore(threshold=3)  # 查看异常
     .get_summary())

    res_1 = preprocessor.get_processed_data()
    history = preprocessor.get_history()
    print("\n处理历史:")
    print(history)

    # 使用重采样类
    resampler = DataResampler(res_1)
    res_2 = (resampler
             .systematic_resample(start_index=5, step=6)  # 切片，从第一小时开始（索引5开始），每隔6个记录一次
             .get_summary()
             .get_resampled_data())

    # 异常值处理类
    handler = ExtremeDataHandler(res_2)
    res_3 = (handler.create_extreme_features_zscore(threshold=3)  # 取样后的异常查看
             .custom_handler()  # 物理异常'wv', 'max. wv'
             .get_handled_data())

    # cnn_lstm 特殊处理类
    special_preprocessor = SpecialCnnLstm(res_3)
    special_preprocessor.prepare_for_cnn_lstm()  # ['Date Time','wv','max. wv','wd']

    """划分数据集"""
    # 提取标签列和特征列
    processed_data = special_preprocessor.get_special_processed_data()
    label_data = processed_data[['T', 'p']]
    feature_data = processed_data.drop(['T', 'p'], axis=1)

    # 三分数据集（70%训练集，20%验证集，10%测试集）
    splitter = DataSplitter(special_preprocessor)
    (splitter.train_val_test_split(feature_data, label_data, train_size=0.7, val_size=0.2)
     .standardize_data())

    # 画小提琴图（观察整个df标准化后的数据分布)
    df1_std = pd.concat([splitter.X_train, splitter.X_val, splitter.X_test], ignore_index=False)
    df2_std = pd.concat([splitter.y_train, splitter.y_val, splitter.y_test], ignore_index=False)
    df_std = df1_std.join(df2_std, how='inner')
    df_std = df_std.melt(var_name='Column', value_name='Standardized')  # 宽表变长表，数据形状匹配

    viz = Visualization()
    viz.violin_plot(df=df_std,
                    var_name='Column', value_name='Standardized',
                    title="统计分布小提琴图")

    """构建窗口数据"""
    # 1 使用WindowGenerator类实例 构造窗口数据
    df_train = pd.concat([splitter.X_train, splitter.y_train], axis=1)
    df_val = pd.concat([splitter.X_val, splitter.y_val], axis=1)
    df_test = pd.concat([splitter.X_test, splitter.y_test], axis=1)

    # 指定预测特征列
    single_window = WindowGenerator(input_width=6, label_width=5, shift=24, label_columns=['T', 'p'],
                                    train_df=df_train, val_df=df_val, test_df=df_test)
    print(single_window)  # 实例直接打印？

    # 未指定预测特征列，预测所有列
    multi_window = WindowGenerator(input_width=6, label_width=5, shift=24,
                                train_df=df_train, val_df=df_val, test_df=df_test)
    print(multi_window)

    # 2 构建训练集、验证集和测试集
    print('训练数据：')
    print(single_window.createTrainSet)
    print('验证数据：')
    print(single_window.createValSet)
    print('测试数据：')
    print(single_window.createValSet)

    # 从数据集中获取输入输出形状元组
    for train_inputs, train_labels in single_window.createTrainSet.take(1):
        print(f'Inputs shape (batch,time,features):{train_inputs.shape}')
        print(f'Labels shape (bathc,time,features):{train_labels.shape}')

    """=========================================CNN========================================="""
    """构建并编译CNN模型"""
    # 基于历史6个时间点的天气情况（6行19列）预测经过24小时（shift=24)未来5个时间点 'T''p'列
    timeseries_cnn_model = CnnModel(architecture_type='parallel') # 分支并行模式
    timeseries_cnn_model._build_parallel_model(input_shape=train_inputs.shape,
                                               output_shape=train_labels.shape,
                                               branch_filters =[[32,32],[64,64]],
                                               branch_kernels=[[2,3],[2,3]], # 2个分支
                                               branch_dilation_rate=[[1,1],[1,1]], # 2个分支都取默认1
                                               activation='relu') # swish
    """训练CNN模型"""
    history_cnn,best_model_cnn = TrainingCnn(model=timeseries_cnn_model,window=single_window)
    timeseries_cnn_model.summary()  # 出来一个表 显示每一层参数个数

    """评估CNN模型"""
    val_mae_cnn,test_mae_cnn=evaluate_model(name='conv',model=best_model_cnn,window=single_window)
    print(f"评估模型 best_model_cnn 的验证集和测试集的均方绝对值误差MAE结果如下：")
    print(val_mae_cnn,test_mae_cnn)

    """=========================================LSTM========================================="""
    """构建并编译LSTM模型1"""
    timeseries_lstm1_model=LstmModel()
    timeseries_lstm1_model._build_sequential_model(
                                        units=[64,],
                                        return_sequences=[False,], # 只输出最后一行
                                        output_shape=train_labels.shape )
    """训练LSTM模型1"""
    history_lstm1,best_model_lstm1= TrainingLstm(model=timeseries_lstm1_model,window=single_window,
                                           file_path='best_model_lstm1.h5')
    timeseries_lstm1_model.summary() # 参数个数
    """评估LSTM模型1"""
    val_mae_lstm1,test_mae_lstm1=evaluate_model(name ='lstm1',model=best_model_lstm1,window=single_window)
    print(f"评估模型 best_model_lstm1 的验证集和测试集的均方绝对值误差MAE结果如下：")
    print(val_mae_lstm1,test_mae_lstm1)

    """LSTM模型2"""
    timeseries_lstm2_model = LstmModel()
    timeseries_lstm2_model._build_sequential_model(
        units=[64,64], # 2层LSTM
        return_sequences=[True,False],  # 只输出最后一行
        output_shape=train_labels.shape)
    """训练LSTM模型2"""
    history_lstm2, best_model_lstm2 = TrainingLstm(model=timeseries_lstm2_model, window=single_window,
                                                   file_path='best_model_lstm2.h5')
    timeseries_lstm2_model.summary()  # 参数个数
    """评估LSTM模型2"""
    val_mae_lstm2,test_mae_lstm2=evaluate_model(name='lstm2',model=best_model_lstm2, window=single_window, )
    print(f"评估模型 best_model_lstm2 的验证集和测试集的均方绝对值误差MAE结果如下：")
    print(val_mae_lstm2,test_mae_lstm2)


    """=========================================比较CNN和LSTM的预测效果========================================="""
    # 画出每个模型里面测试集和验证集的MAE
    val_mae =[val_mae_cnn,val_mae_lstm1,val_mae_lstm2]
    test_mae = [test_mae_cnn,test_mae_lstm1,test_mae_lstm2]
    x = len(val_mae) # 3个模型

    plt.ylabel('mean_absolute_error')  # 指定纵轴标签
    plt.bar(x=x-0.17, height=val_mae, width=0.3, label='Validation')
    plt.bar(x=x+0.17, height=test_mae, width=0.3, label='Test')
    plt.xticks(ticks=x, labels=['conv1D','lstm1','lstm2'],rotation=45)
    _ = plt.legend()




if __name__ == "__main__":
    main()
