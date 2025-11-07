# from src.data.exploration import Visualization

# from src.models.cnn import CnnModel
# from src.models.lstm import LstmModel
# from src.training.training_models import TrainingModel
# from src.evaluation.metrics import evaluate_model
# from .trained.trained import ReconstructPredictor
# import matplotlib.pyplot as plt
# from src.utils.debug_controller import DebugController
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


from models.NeuralNetwork import TimeSeriesEstimator
from data.data_preprocessing import SystematicResampler

# 优化方向：
# 追加参数检测 window / handle_extre_numeric / check_extre 异常值可以针对每列选择不同填充方式 或 改写自定义只处理单列
# process_categorical_cols 支持排除列
# 时间列处理中新生成列 独立放入feature_generator
# 每种类型里面 生成新的特征 模块点 小标题系统点（风矢量是数值列、process将encoding onehot 纳入、feature generator 可以独立、时间列（单）处理和生成分离

from src.utils.config import TensorFlowConfig
from pipelines.preprocess_pipeline import CompletePreprocessor
from data.data_preprocessing import TimeSeriesSplitter
from data.data_preparation import (DataLoader, DescribeData, RemoveDuplicates, DeleteUselessCols, ProblemColumnsFixed,
                                   SpecialColumnsFixed, CheckExtreFeatures,
                                   MarkNumericOutlier,
                                   CategoricalOutlierProcessor, NumericMissingValueHandler,
                                   CategoricalMissingValueHandler,
                                   ColumnsTypeIdentify,
                                   ConvertCategoricalColumns,
                                   ConvertNumericColumns)

from data.feature_engineering import (GenerationFromNumeric, ProcessTimeseriesColumns, BasedOnCorrSelector,
                                      UnifiedFeatureScaler, CategoricalEncoding)
from data.windows import WindowGeneratorForNeural
from data.exploration import VisualizationForNeural

def main():
    TensorFlowConfig.setup_environment()
    '''
    1. 严格按照数据的【处理顺序】使用‘class’，并标记'len_change'(这里将改变数据长度的步骤，手动处理）
    2. 手动处理的类:是无法放进pipeline的类，不会继承BaseEstimator和TransfromerMixin。并且使用learn_process处理。
    '''
    check_outliers_config = {'method': 'iqr', 'threshold': 1.5}

    download_outliers_details_config = {
        'enabled': True,
        'path': '~/Python/NeuralNetwork/temperature_forecasting/data/intermediate',
        'filename': 'outliers.csv'}

    numeric_outliers_config = [
        ('zscore', {'threshold': 3, 'columns': ['T', 'Tpot', 'Tdew']}),  # 前额常接近正态分布，Z-score效果好
        ('iqr', {'threshold': 1.5, 'columns': ['p', 'VPmax', 'VPact', 'VPdef']}),  # 气压有明确的物理范围，IQR对中等离群值敏感
        ('robust', {'quantile_range': (5, 95), 'columns': ['rh', 'sh', 'H2OC']}),  # 分位数检测对分布偏斜
        ('isolationforest', {'contamination': 0.025, 'columns': ['rho', 'wv', 'max. wv', 'wd']})  # 对复杂分布效果好
    ]
    numeric_missing_config = {
        'spec_fill': [
            {'constant': {'columns': ['A', ], 'fill_value': [30, ]}},  # 区分scaler，这里是按顺序，后已调整
            {'mode': {'columns': ['C', 'D', 'E']}}
        ],
        'skip_fill': ['max. wv', 'wv'],
        'smart_fill_remain': True,
        'important_columns': ['T', 'p'],
    }
    scaling_config = {
        'transformers': [
            {'minmax': {'columns': ['T', 'S'], 'feature_range': (0, 1),...}},  # 相同方法，相同其他参数配置，在columns列表填写
            {'minmax': {'columns': ['W'], 'feature_range': (-1, 1)}},  # 相同方法，但是其他参数配置与前一配置不同，允许在下一行填写
            {'standard': {'columns': ['P', 'Q']}},
            {'robust': {'columns': ['R'], 'quantile_range': (10, 90)}}
        ],
        'skip_scale': ['is_night']  # 跳过二分类列
    }

    window_config = {
        'input_width': 6, 'label_width': 5, 'shift': 24, 'label_columns':['T','p']
    }


    preparation_configs = [
        {'class': [DescribeData()], 'len_change': False},
        {'class': [RemoveDuplicates(), DeleteUselessCols()], 'len_change': True},
        {'class': [ColumnsTypeIdentify(),
                   ConvertCategoricalColumns(), ConvertNumericColumns(preserve_integer_types=True, exclude_cols=[]),
                   ProblemColumnsFixed(problem_columns=['wv']), SpecialColumnsFixed(problem_columns=['T']),
                   CheckExtreFeatures(method_config=check_outliers_config,
                                      download_config=download_outliers_details_config),
                   MarkNumericOutlier(method_config=numeric_outliers_config),
                   CategoricalOutlierProcessor(pass_through=True),
                   NumericMissingValueHandler(method_config=numeric_missing_config),
                   CategoricalMissingValueHandler(method_config=None, pass_through=True)], 'len_change': False},

        {'class': [SystematicResampler(start_index=5, step=6, reset_index=True)], 'len_change': True},

        {'class': [GenerationFromNumeric(dir_cols=['wd'], var_cols=['wv', 'max. wv']),
                   ProcessTimeseriesColumns(interactive=True, auto_detect_string_format=True),  # 开启交互式功能 + 自动检测时间列
                   BasedOnCorrSelector(pass_through=True),
                   UnifiedFeatureScaler(method_config=None, algorithm='cnn'),  # 自动根据数据分布及算法类型进行推荐标准化
                   CategoricalEncoding(handle_unknown='ignore', unknown_token='__UNKNOWN__'),
                   ], 'len_change': False},
        {'class':[VisualizationForNeural()],'len_change':[False]},
        {'class':[WindowGeneratorForNeural(**window_config)], 'len_change':[True]}
    ]

    # 1. 加载数据
    loader = DataLoader(input_files=['data_climate'], pattern="new_*.csv", data_dir="'data'/'raw'")
    raw_data = loader.learn_process()

    # 2. 数据集分割
    splitter = TimeSeriesSplitter(train_size=0.6, val_size=0.3, test_size=0.1, shuffle=False)
    df_train, df_val, df_test = splitter.learn_process(raw_data)

    # 3. 数据预处理
    preparation = CompletePreprocessor(preparation_configs)
    features_temp_train = preparation.train(features=df_train, labels=None) # 窗口输出inputs,labels
    features_temp_val = preparation.transform(features=df_val, labels=None)
    features_temp_test = preparation.transform(features=df_test, labels=None)

    stage = preparation.pipelines.get('pipeline_5') # 取第5个class里面的pipeline_5
    cat_cols = stage.named_steps['engineer_4'].categorical_columns_ # 第 4step 的scaler
    num_cols = stage.named_steps['engineer_5'].numeric_columns_ # 第 5step 的encoding



    # 4. 并行模型训练（需要+ val输入，参与模型训练）


    preprocessor.train(raw_data, labels=None)

    # 4. 预测

    predictions = preprocessor.predict(new_data)

    print(f"生成 {len(predictions)} 个预测结果")
    return predictions


# .train_val_test_split(train_size=0.7, val_size=0.2, test_size=0.1)
# .unify_feature_scaling(transformers=config))  # 独热编码 / 分类型 / 时间不处理
# .systematic_resample(start_index=5, step=6)  # 切片，从第一小时开始（索引5开始），每隔6个(6*10分钟)采一次


config = [
    ('minmax', {'feature_range': (0, 1), 'columns': ['T']}),
    # ('std_scaler', {'columns': []}),
    # ('robust_scaler', {'quantile_range': (25, 75), 'columns': []})
]

if __name__ == "__main__":
    main()


# """构建窗口数据"""
# # 1 使用WindowGenerator类实例 构造窗口数据
# df_train = preprocessor.get_train_val_test_data()[0]
# df_val = preprocessor.get_train_val_test_data()[1]
# df_test = preprocessor.get_train_val_test_data()[2]

# # 指定预测特征列
# single_window = WindowGenerator(input_width=6, label_width=5, shift=24, label_columns=['T', 'p'])
# print(single_window)

# 2 构建训练集、验证集和测试集
# print('训练数据：')
# window_train_data = single_window.createDataset(df_train)
# print(window_train_data)
# 正确获取inputs和labels的形状
# createTrainSet()返回的是tf.data.Dataset对象，不是普通的元组列表
# 必须使用TensorFlow的迭代机制：iter(), take()正确解包inputs和labels ：for... .take(1) / example有iter
# train_inputs, train_labels = single_window.example
# print(f"train_inputs 形状: {train_inputs.shape}")
# print(f"train_labels 形状: {train_labels.shape}")

# print('验证数据：')
# window_val_data = single_window.createDataset(df_val)
# print(window_val_data)
# print('测试数据：')
# window_test_data = single_window.createDataset(df_test)
# print(window_test_data)

# # 画个训练集的图
# single_window.window_plot(plot_col='T')

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

# 画出每个模型里面测试集和验证集的MAE
val_mae = [val_mae_cnn, val_mae_lstm1, val_mae_lstm2]
test_mae = [test_mae_cnn, test_mae_lstm1, test_mae_lstm2]
x = len(val_mae)  # 3个模型

plt.ylabel('mean_absolute_error')  # 指定纵轴标签
plt.bar(x=x - 0.17, height=val_mae, width=0.3, label='Validation')
plt.bar(x=x + 0.17, height=test_mae, width=0.3, label='Test')
plt.xticks(ticks=x, labels=['conv1D', 'lstm1', 'lstm2'], rotation=45)
_ = plt.legend()
