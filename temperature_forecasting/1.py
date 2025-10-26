# class SmartScalerSelector:
#     """智能标准化方法选择器，
#        暂未支持添加skip列表"""
#
#     def __init__(self, zscore_outlier_threshold=3.0, skewness_threshold=2.0,
#                  iqr_outlier_threshold=1.5, ):  # abs(skewness) > 1:  严重偏态 对数变换
#
#         self.zscore_outlier_threshold = zscore_outlier_threshold
#         self.skewness_threshold = skewness_threshold
#         self.iqr_outlier_threshold = iqr_outlier_threshold
#         self.scaling_recommendations = {}  # 存储标准化处理建议，理由，及stats_info
#
#     def analyze_feature(self, series):
#         """分析单个特征的统计特性"""
#         series_clean = series.dropna()
#         n_unique = series_clean.nunique()
#
#         # if n_unique < 2:  # 常数列检测
#         #     return {
#         #         'is_constant': True,
#         #         'n_samples': len(series_clean),
#         #         'n_unique': n_unique,
#         #         'constant_value': series_clean.iloc[0] if len(series_clean) > 0 else 0,
#         #     }
#         # 正常的基本统计量
#         stats_info = {
#             'is_constant': False,
#             'n_samples': len(series_clean),
#             'n_unique': n_unique,
#             'mean': np.mean(series_clean),
#             'median': np.median(series_clean),  # iqr 要求4个以上，中位数2个都行
#             'std': np.std(series_clean),
#             'min': np.min(series_clean),
#             'max': np.max(series_clean),
#             'range': np.max(series_clean) - np.min(series_clean),
#         }
#
#         # 只有数据量足够时才计算高阶矩
#         if len(series_clean) >= 4:
#             stats_info.update({
#                 'skewness': stats.skew(series_clean),
#                 'kurtosis': stats.kurtosis(series_clean)
#             })
#         if stats_info.get('std', 0) < 1e-8:
#             stats_info['is_near_constant'] = True
#             stats_info['constant_value'] = series_clean.iloc[0]
#
#         # 异常值检测
#         # zscore
#         z_score = np.abs(stats.zscore(series_clean))
#         outlier_count_z = np.sum(z_score > self.zscore_outlier_threshold)
#         outlier_ratio_z = outlier_count_z / len(series_clean)
#
#         # iqr
#         q1, q3 = np.percentile(series_clean, [25, 75])
#         iqr = q3 - q1
#         lower_bound = q1 - iqr * self.iqr_outlier_threshold
#         upper_bound = q3 + iqr * self.iqr_outlier_threshold
#         outlier_count_iqr = np.sum((series_clean < lower_bound) | (series_clean > upper_bound))
#         outlier_ratio_iqr = outlier_count_iqr / len(series_clean)
#
#         if len(series_clean) <= 4 or series_clean.nunique() <= 4:
#             stats_info.update({
#                 'outlier_ratio_z': outlier_ratio_z,
#                 'recommendation': 'standard',
#                 'reason': '数据唯一值<=4或数据<=4不足，不推荐robust标准化，可尝试Z-score标准化'
#             })
#
#         else:
#             stats_info.update({
#                 'outlier_ratio_z': outlier_ratio_z,
#                 'outlier_ratio_iqr': outlier_ratio_iqr,
#                 'iqr': iqr
#             })
#
#         return stats_info
#
#     def recommend_scaler(self, stats_info, method=None):
#         """基于统计特征推荐标准化方法"""
#         if stats_info.get('is_constant', False) or stats_info.get('is_near_constant', False):
#             constant_value = stats_info.get('constant_value', 0)
#
#             # 根据原始方式决定常数列的处理方式
#             if method == 'minmax':
#                 return 'constant_minmax', f"常数列值{constant_value}，{method}后设为0.5"
#             elif method == 'standard':
#                 return 'constant_standard', f"常数列值{constant_value}，{method}后设为0"
#             elif method == 'robust':
#                 return 'constant_robust', f"常数列值{constant_value}，{method}后设为0"
#             else:
#                 return 'constant_standard', f"常数列值{constant_value}，{method}后设为0"
#
#         # 非常数列
#         large_range = stats_info.get('range', 0) > 1000
#         high_outliers = (stats_info.get('outlier_ratio_z', 0) > 0.05) or (stats_info.get('outlier_ratio_iqr', 0) > 0.05)
#         high_skewness = np.abs(stats_info.get('skewness', 0)) > self.skewness_threshold
#
#         if high_outliers:
#             if high_skewness:
#                 return 'robust', '存在异常值且分布偏斜，推荐鲁棒标准化'
#             else:
#                 return 'robust', '存在异常值但分布相对正常，推荐鲁棒标准化'
#         elif high_skewness:
#             return 'minmax', '分布偏斜但异常值较少，推荐MinMax归一化'
#         elif large_range:
#             return 'standard', '数值范围大但分布相对正常，推荐Z-score标准化'
#         else:
#             return 'standard', '分布相对正常、数据范围相对正常、异常值较少，推荐Z-score标准化'  # 暂无不标准化的逻辑
#
#     def process(self, df, col=None):
#         """分析数据集中的所有数值列"""
#         if col is None:
#             numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
#         else:
#             numeric_cols = list(col)
#
#         for col in numeric_cols:
#             stats_info = self.analyze_feature(df[col])
#
#             if stats_info.get('recommedation', None) == 'skip':  # 样本<=2
#                 self.scaling_recommendations[col] = {
#                     'scaler': 'skip',
#                     'reason': stats_info['reason'],
#                 }
#             elif stats_info.get('recommedation', None) == 'standard':  # 样本<=4
#                 self.scaling_recommendations[col] = {
#                     'scaler': 'standard',
#                     'reason': stats_info['reason'],
#                     'stats': stats_info
#                 }
#             else:
#                 scaler_type, reason = self.recommend_scaler(stats_info)
#                 self.scaling_recommendations[col] = {
#                     'scaler': scaler_type,
#                     'reason': reason,
#                     'stats': stats_info
#                 }
#
#         return self
#
#     def get_recommendations(self):
#         """获取标准化推荐结果"""
#         return self.scaling_recommendations
#
#     def print_reconmendations(self):
#         print("=== 智能标准化方法推荐 ===")
#         for col, info in self.scaling_recommendations.items():
#             stats = info['stats']
#             print(f"\n列: {col}")
#             print(f"  推荐方法: {info['scaler']}")
#             print(f"  原因: {info['reason']}")
#             print(
#                 f"  样本数: {stats['n_samples']}, 唯一样本：{stats['n_unique']},范围: [{stats.get('min', 0):.2f}, {stats.get('max', 0):.2f}]")
#             print(
#                 f"  均值: {stats.get('mean'):.3f}, 标准差: {stats.get('std'):.3f}, 常数列:{stats.get('is_constant', False)},近似常数列：{stats.get('is_near_constant', False)}")
#             print(
#                 f"  偏度: {stats.get('skewness', 0):.3f}, 异常值比例(Z-score): {stats.get('outlier_ratio_z', 0):.3f},异常值比例(iqr):{stats.get('outlier_ration_iqr'):.3f}")
#
#
# class AlgorithmAwareScalerSelector:
#     """基于算法特性的标准化选择器：处理单列/不处理常数列"""
#
#     def __init__(self):
#         # 1.标准化要求 强->弱 ；2.nunique要求 高->低
#
#         self.algorithm_requirements = {
#             # 神经网络
#             'neural_network': {
#                 'priority': ['standard', 'minmax'],
#                 'standardization': 'required',
#                 'reason': '梯度下降需要特征尺度一致',
#                 'min_nunique': 2,
#                 'min_samples': 10  # 神经网络需要足够数据避免过拟合
#             },
#             'cnn': {
#                 'priority': ['minmax', 'standard'],
#                 'standardization': 'required',
#                 'reason': '卷积核和激活函数在归一化范围效果更好',  # CNN通常更偏好[0,1]范围
#                 'min_nunique': 2,
#                 'min_samples': 20
#             },
#             'lstm': {
#                 'priority': ['standard', 'minmax'],
#                 'standardization': 'required',
#                 'reason': '门控机制对输入分布的均值和方差更敏感',
#                 'min_nunique': 2,
#                 'min_samples': 15
#             },  # zscore优先
#
#             # 基于距离的算法（基于距离的算法还需要标准差） nunique >= 2
#             'knn': {
#                 'priority': ['standard', 'minmax'],
#                 'standardization': 'required',
#                 'reason': '对特征尺度敏感',
#                 'min_nunique': 2,
#                 'min_samples': 5  # 至少需要几个邻居
#             },
#             'kmeans': {
#                 'priority': ['standard', 'minmax'],
#                 'standardization': 'required',
#                 'reason': '基于距离的聚类',
#                 'min_nunique': 2,
#                 'min_samples': 3,  # 至少比聚类中心数多
#             },
#             'svm': {
#                 'priority': ['standard', 'robust'],
#                 'standardization': 'required',
#                 'reason': '依赖特征距离',
#                 'min_nunique': 2,
#                 'min_samples': 2,  # 至少需要支持向量
#             },
#             'dbscan': {
#                 'priority': ['standard', 'robust'],
#                 'standardization': 'required',
#                 'reason': '基于密度和距离的聚类'
#             },
#
#             # 线性模型 推荐无异常值 nunique>= 2
#             'linear_regression': {
#                 'priority': ['standard', 'robust'],
#                 'standardization': 'required',
#                 'reason': '系数解释需要标准化',
#                 'min_nunique': 2,
#                 'min_samples': 3,  # 至少n+1个样本(n是特征数)
#             },
#             'logistic_regression': {
#                 'priority': ['standard', 'robust'],
#                 'standardization': 'required',
#                 'reason': '收敛速度和系数解释',
#                 'min_nunique': 2,
#                 'min_samples': 10,  # 需要足够样本保证类别平衡
#             },
#             'pca': {
#                 'priority': ['standard', 'robust'],
#                 'standardization': 'required',
#                 'reason': '主成分分析对方差敏感',
#                 'min_nunique': 2,
#                 'min_samples': 5,  # 需要足够样本估计协方差
#             },
#             'gaussian_mixture': {
#                 'priority': ['standard', 'robust'],
#                 'standardization': 'required',
#                 'reason': '高斯混合模型对特征尺度敏感，需要统一方差',
#                 'min_nunique': 2,
#                 'min_samples': 10,  # 高斯分布需要足够样本估计参数
#             },
#
#             # 树模型 nunique >= 1 ，标准化推荐但不必须,虽然能用 nunique = 1 的数据，但这样的特征在模型中会被忽略（无法分裂）
#             'random_forest': {
#                 'priority': ['minmax', 'standard'],
#                 'standardization': 'recommended',
#                 'reason': '对尺度不敏感但标准化可加速收敛',
#                 'min_nunique': 1,  # 树模型能用常数列！
#                 'min_samples': 2,  # 至少需要分裂一次
#             },
#             'xgboost': {
#                 'priority': ['minmax', 'standard'],
#                 'standardization': 'recommended',
#                 'reason': '内置特征缩放但仍可从外部标准化受益',
#                 'min_nunique': 1,  # 也能处理常数列
#                 'min_samples': 2,
#             },
#             'lightgbm': {
#                 'priority': ['minmax', 'standard'],
#                 'standardization': 'recommended',
#                 'reason': '类似XGBoost',
#                 'min_nunique': 1,  # 同样处理常数列
#                 'min_samples': 2,
#             },
#             'gradient_boosting': {
#                 'priority': ['minmax', 'standard'],
#                 'standardization': 'recommended',
#                 'reason': '树模型集成，标准化可改善收敛',
#                 'min_nunique': 1,  # 能处理常数列，但这样的特征会被忽略
#                 'min_samples': 2,  # 至少需要2个样本计算梯度
#             },
#
#             # 概率模型 (不需要或轻度需要标准化)
#             'gaussian_naive_bayes': {
#                 'priority': ['standard', 'none'],
#                 'standardization': 'recommended',
#                 'reason': '高斯假设下标准化可使分布更合理',
#                 'min_nunique': 1,  # 常数列也能用
#                 'min_samples': 2,  # 需要估计均值和方差
#             },  # 如果要用标准化，standard优先。推荐但不是必须
#
#             'naive_bayes': {
#                 'priority': ['none', 'standard'],
#                 'standardization': 'not_needed',
#                 'reason': '概率模型，对特征尺度不敏感',
#                 'min_nunique': 1,  # 常数列也能计算概率
#                 'min_samples': 1,  # 理论上1个样本就能工作
#             },
#
#             'decision_tree': {
#                 'priority': ['none', 'minmax'],
#                 'standardization': 'not_needed',
#                 'reason': '尺度无关，基于值排序分裂',
#                 'min_nunique': 1,  # 常数列也能用（只是无法分裂）
#                 'min_samples': 1,  # 理论上1个样本就能建树
#             },  # 优先级1: 不处理，优先级2: 如有需要用minmax
#
#         }
#
#     def recommend_for_algorithm(self, algorithm, feature_stats):
#         if feature_stats.get('n_samples', 0) < 2:  # feature_stats 是SmartScalerSelector里面的stats_info
#             return None, '数据不足'
#         # 规则算法 Decision Tree / 朴素贝叶斯 要求nunique>=1（常数列可）需要调整， 其他2
#
#         if algorithm not in self.algorithm_requirements:
#             return 'standard', '不在已设置的算法内，使用Z-score标准化'
#
#         priorities = self.algorithm_requirements[algorithm]['priority']
#
#         # 基于数据特征调整推荐(需要加）
#         if feature_stats.get('outlier_ratio_iqr', 0) > 0.05 and 'robust' in priorities:
#             return 'robust', f"存在异常值，为{algorithm}推荐鲁邦标准化"
#         elif feature_stats.get('skewness', 0) > 2.0 and 'minmax' in priorities:
#             return 'minmax', f"分布偏斜，为{algorithm}推荐MinMax归一化"
#         else:
#             return priorities[0], f"为{algorithm}推荐{priorities[0]}"