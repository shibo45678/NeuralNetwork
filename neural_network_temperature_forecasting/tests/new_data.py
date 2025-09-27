# 预测数据的预处理（完成标准化）

input_files
preprocessor  = (DataPreprocessor(input_files = ["data_climate.csv"])
                        .handleEncoding() # 编码utf-8
                        .load_all_data(pattern="new_*.csv") # 加载
                        .describe_data()
                        .problem_columns_fixed(problem_columns =[])
                        .special_columns_fixed(problem_columns=['T'])
                        .identify_column_types()
                        .process_numeric_data()
                        .encode_categorical_data()
                        .process_other_data()
                        .handle_missing_values(cat_strategy='mode', num_strategy='median')
                        .remove_duplicates()
                        .delete_useless_cols(target_cols=None)
                        .check_extreme_features({'name':'iqr','threshold':1.5}) # 查看
                        .check_extreme_features({'name':'zscore','threshold':3})
                        .check_extreme_features({'name':'multivariate','contamination':0.025}) # 预期异常比例
                        .systematic_resample(start_index=5, step=6) # 切片，从第一小时开始（索引5开始），每隔6个(6*10分钟)采一次
                        .remove_outliers(method='custom') # 目前仅处理了少数物理异常
                        .handle_time_col(col='Date Time',format='%d.%m.%Y %H:%M:%S') # time_col正余弦
                        .handle_vec_col(dir_cols=['wd'],var_cols=['wv','max. wv']) # vec_col 风矢量 要求顺序
                        )
preprocessor.get_data()
