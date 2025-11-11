# 优化方向：
# 追加参数检测 window / handle_extre_numeric
# process_categorical_cols 支持排除列
# 时间列处理中新生成列 独立放入feature_generator
# 每种类型里面 生成新的特征 模块点 小标题系统点（风矢量是数值列、process将encoding onehot 纳入、feature generator 可以独立、时间列（单）处理和生成分离
# 模型output_config是单列配置，优化成同一类配置可以一起跑。什么都不写默认回归？分类？
# 缺失值填充 增加算法填充等
# missing列 从提示改为报错 尽早暴露/结果无意义/
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from models.NeuralNetwork import TimeSeriesEstimator
from src.utils.tensorflow_config import TensorFlowConfig
from pipelines.preprocess_pipeline import CompletePreprocessor
from data.data_preprocessing import TimeSeriesSplitter
from data.data_preparation import (DataLoader, DescribeData, RemoveDuplicates, DeleteUselessCols, ProblemColumnsFixed,
                                   SpecialColumnsFixed, CheckExtreFeatures,
                                   CustomNumericOutlier, handler,
                                   CategoricalOutlierProcessor, NumericMissingValueHandler,
                                   CategoricalMissingValueHandler,
                                   ColumnsTypeIdentify,
                                   ConvertCategoricalColumns,
                                   ConvertNumericColumns)
from data.data_preprocessing import SystematicResampler
from data.feature_engineering import (GenerationFromNumeric, ProcessTimeseriesColumns, BasedOnCorrSelector,
                                      UnifiedFeatureScaler, CategoricalEncoding)
from data.exploration import VisualizationForNeural
import logging.config
import logging_config

def main():
    logging.config.dictConfig(logging_config.LOGGING_CONFIG)
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
        ('isolationforest', {'contamination': 0.025, 'columns': ['rho', 'wd']}),  # 对复杂分布效果好
        ('custom', {'functions': [handler(series=None)], 'columns': ['wv', 'max. wv']})
    ]
    numeric_handle_config = {'custom': ['wv', 'max. wv']}

    numeric_missing_config = {
        'spec_fill': [
            {'constant': {'columns': ['wv', 'max. wv'], 'fill_value': [0, 0]}},  # 区分scaler，这里是按顺序，后已调整
            {'mode': {'columns': []}},
            {'ffill': {'columns': ['p']}},  # 气压变化相对连续稳定，短期内有持续性
            {'bfill': {'columns': ['rh']}}  # 湿度变化相对缓慢，受天气系统影响有持续性
        ],
        'skip_fill': ['max. wv', 'wv'],
        'smart_fill_remain': True,
        'important_columns': ['T', 'p'],
    }
    # 新特征生成后
    scaling_config = {
        'transformers': [
            {'standard': {
                'columns': ['T', 'p', 'Tpot', 'Tdew', 'wv', 'max. wv_y', 'wv_x', 'wv_y', 'max. wv_x', 'max. wv_y']}},
            {'minmax': {'columns': ['rh', 'VPmax', 'Vpact', 'VPdef', 'sh', 'H2OC', 'rho'], 'feature_range': (0, 1)}},
            # 相同方法，相同其他参数配置，在columns列表填写
            {'minmax': {'columns': [], 'feature_range': (-1, 1)}},  # 相同方法，但是其他参数配置与前一配置不同，允许在下一行填写
            {'robust': {'columns': [], 'quantile_range': (10, 90)}}
        ],
        'skip_scale': ['is_night']  # 跳过二分类列(数值型）/ 异常值标记列自动skip
    }

    output_config = {
        'T': {'type': 'regression',  # 单变量回归
              'loss': 'mse',
              'metrics': ['mae'],
              'units': 1,  # 每个时间步预测n个特征
              },

        'p': {'type': 'regression',
              'loss': 'mse',
              'metrics': ['mae'],
              'units': 1,
              }
    }

    preparation_configs = [
        {'class': [DescribeData()], 'len_change': False},
        {'class': [RemoveDuplicates(), DeleteUselessCols()], 'len_change': True},
        {'class': [ColumnsTypeIdentify(),
                   ConvertCategoricalColumns(), ConvertNumericColumns(preserve_integer_types=True, exclude_cols=['Data Time']),
                   ProblemColumnsFixed(problem_columns=['wv']), SpecialColumnsFixed(problem_columns=['T']),
                   CheckExtreFeatures(method_config=check_outliers_config,
                                      download_config=download_outliers_details_config),
                   CustomNumericOutlier(method_config=numeric_outliers_config, handle_config=numeric_handle_config),
                   CategoricalOutlierProcessor(pass_through=True),
                   NumericMissingValueHandler(method_config=numeric_missing_config),
                   CategoricalMissingValueHandler(method_config=None, pass_through=True)], 'len_change': False},

        {'class': [SystematicResampler(start_index=5, step=6, reset_index=True)], 'len_change': True},

        {'class': [GenerationFromNumeric(dir_cols=['wd'], var_cols=['wv', 'max. wv']),
                   ProcessTimeseriesColumns(interactive=True, auto_detect_string_format=True),  # 开启交互式功能 + 自动检测时间列
                   BasedOnCorrSelector(pass_through=True),
                   UnifiedFeatureScaler(method_config=scaling_config, algorithm='lstm'),  # 自动根据数据分布及算法类型进行推荐标准化
                   CategoricalEncoding(handle_unknown='ignore', unknown_token='__UNKNOWN__'),
                   ], 'len_change': False},
        {'class': [VisualizationForNeural()], 'len_change': [False]},
    ]

    # 1. 加载数据
    loader = DataLoader(input_files=['data_climate'], pattern="new_*.csv", data_dir="'data'/'raw'")
    raw_data = loader.learn_process()

    # 2. 数据集分割
    splitter = TimeSeriesSplitter(train_size=0.6, val_size=0.3, test_size=0.1, shuffle=False)
    df_train, df_val, df_test = splitter.learn_process(raw_data)

    # 3. 数据预处理(生成训练、验证、预测数据）
    preprocessor = CompletePreprocessor(preparation_configs)
    features_temp_train = preprocessor.train(features=df_train, labels=None)
    features_temp_val = preprocessor.transform(features=df_val, labels=None)
    features_temp_test = preprocessor.transform(features=df_test, labels=None)

    num_cols = preprocessor.pipelines.get_specific_attribute(5, 'engineer_4', 'numeric_columns_')  # 取第5个class的第4步的属性
    cat_cols = preprocessor.pipelines.get_specific_attribute(5, 'engineer_5', 'categorical_columns_')
    time_col = preprocessor.pipelines.get_specific_attribute(5, 'engineer_2', 'time_column_')

    # 4. 并行模型训练、评估
    base_model_config = {'numeric_columns': num_cols,
                         'categorical_columns': cat_cols,
                         'time_column': time_col,
                         'input_width': 6,
                         'label_width': 5,
                         'shift': 24,
                         'label_columns': ['T', 'p'],
                         'output_config': output_config,
                         }

    cnn_model_config = {**base_model_config, **{
        'model_type': 'cnn',
        'branch_filters': [[32, 32], [64, 64]],
        'branch_kernels': [[2, 3], [2, 3]],
        'branch_dilation_rate': [[1, 1], [1, 1]],
        'activation': 'relu',
        'learning_rate': 0.001,
        'epochs': 20,
        'verbose': 2
    }}

    lstm_model_config1 = {**base_model_config, **{
        'model_type': 'lstm1',
        'learning_rate': 0.001,
        'units': [64],  # len控制lstm的层数
        'return_sequences': [False],
        'epochs': 100,
        'verbose': 2
    }}

    lstm_model_config2 = {**base_model_config, **{
        'model_type': 'lstm2',
        'learning_rate': 0.001,
        'units': [64, 32],  # 逐步压缩特征
        'return_sequences': [True, False],  # 上一轮的输出做本轮输入input + 上一轮输出
        'epochs': 100,
        'verbose': 2
    }}

    data = pd.DataFrame({'train_datasets': features_temp_train, 'val_datasets': features_temp_val}) # 训练要求验证集

    def train_single_config(config, X, y):
        name = config.get('model_type')
        model = TimeSeriesEstimator(config)

        # 训练
        model.fit(X, y=None)
        model.save(f'./saved_{name}/timeseries_v1')

        # 重构、预测
        predictions = model.predict(features_temp_test)  # # N天后加载使用load
        print(f"生成 {len(predictions)} 个预测结果")

        # 逆转换
        inverse_1 = preprocessor.pipelines.get('pipeline_5').named_steps['engineer_4'].inverse_transform(
            scaled_data=predictions, target_columns=['T', 'p'])

        inverse_2 = preprocessor.pipelines.get('pipeline_5').named_steps['engineer_5'].inverse_transform(inverse_1)

        print(f"最终的数据：{inverse_2.head(10)}")

        return inverse_2

    configs = [cnn_model_config, lstm_model_config1, lstm_model_config2]

    failed_configs = []
    trained_models = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(train_single_config, config, data, y=None)
                   for config in configs]

        for future, config in zip(futures, configs):
            try:
                result = future.result()
                trained_models.append(result)
                print("一个模型训练成功")
            except Exception as e:
                print(f"模型{config.get('model_type')}训练失败:{str(e)}")
                failed_configs.append(config)

    print(f"完成: {len(trained_models)} 个成功, {len(failed_configs)} 个失败")
    return trained_models, failed_configs


if __name__ == "__main__":
    results = main()
    setup_logging()

# 画出每个模型里面测试集和验证集的MAE
# val_mae = [val_mae_cnn, val_mae_lstm1, val_mae_lstm2]
# test_mae = [test_mae_cnn, test_mae_lstm1, test_mae_lstm2]
# x = len(val_mae)  # 3个模型
#
# plt.ylabel('mean_absolute_error')  # 指定纵轴标签
# plt.bar(x=x - 0.17, height=val_mae, width=0.3, label='Validation')
# plt.bar(x=x + 0.17, height=test_mae, width=0.3, label='Test')
# plt.xticks(ticks=x, labels=['conv1D', 'lstm1', 'lstm2'], rotation=45)
# _ = plt.legend()
