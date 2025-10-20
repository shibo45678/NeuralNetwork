# 算法需求 → 决定要不要标准化
# nunique >= 2 → 决定能不能标准化
# min_samples → 决定标准化是否可靠
# 统计量计算 → 都受前几步的影响

class a:
    def process_feature_for_algorithm(self,series: pd.Series, algorithm: str) -> Dict[str, Any]:
        """完整的特征处理流程"""

        # 1. 数据基本情况
        series_clean = series.dropna()
        n_unique = series_clean.nunique()
        n_samples = len(series_clean)

        # 2. 统计量计算 (受n_unique和n_samples影响)
        stats_info = calculate_statistics_safely(series)

        # 3. 标准化可行性判断 (受算法需求和数据条件影响)
        standardization_info = can_standardize_feature(series, algorithm)

        # 4. 最终决策
        result = {
            'algorithm': algorithm,
            'basic_stats': stats_info,
            'standardization': standardization_info,
            'final_action': None,
            'final_reason': None
        }

        if standardization_info['can_standardize']:
            result['final_action'] = 'apply_standardization'
            result['final_reason'] = f'使用{standardization_info["recommended_methods"][0]}方法'
        else:
            result['final_action'] = standardization_info['action']
            result['final_reason'] = standardization_info['reason']

        return result


    def calculate_statistics_safely(self,series: pd.Series) -> Dict[str, Any]:
        """安全计算统计量，考虑数据限制"""
        series_clean = series.dropna()
        n_unique = series_clean.nunique()
        n_samples = len(series_clean)

        stats_info = {
            'n_unique': n_unique,
            'n_samples': n_samples,
            'can_calculate_advanced_stats': True
        }

        # 基本统计量 (大部分算法需要)
        stats_info['mean'] = np.mean(series_clean) if n_samples >= 1 else 0
        stats_info['min'] = np.min(series_clean) if n_samples >= 1 else 0
        stats_info['max'] = np.max(series_clean) if n_samples >= 1 else 0

        # 需要至少2个样本的统计量
        if n_samples >= 2:
            stats_info['std'] = np.std(series_clean)
            stats_info['range'] = stats_info['max'] - stats_info['min']
        else:
            stats_info['std'] = 0
            stats_info['range'] = 0
            stats_info['can_calculate_advanced_stats'] = False

        # 需要至少4个样本的高阶统计量
        if n_samples >= 4 and n_unique >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                stats_info['skewness'] = stats.skew(series_clean)
                stats_info['kurtosis'] = stats.kurtosis(series_clean)
        else:
            stats_info['skewness'] = 0
            stats_info['kurtosis'] = 0

        return stats_info

    def can_standardize_feature(series: pd.Series, algorithm: str) -> Dict[str, Any]:
        """判断特征是否能进行标准化"""
        series_clean = series.dropna()
        n_unique = series_clean.nunique()
        n_samples = len(series_clean)

        # 第一步：算法是否需要标准化
        algo_req = self.algorithm_requirements[algorithm]
        standardization_needed = algo_req['standardization'] in ['required', 'recommended']

        if not standardization_needed:
            return {
                'can_standardize': False,
                'reason': f'{algorithm}算法不需要标准化',
                'action': 'keep_original'
            }

        # 第二步：检查标准化可行性 (nunique >= 2)
        if n_unique < 2:
            return {
                'can_standardize': False,
                'reason': '常数列无法标准化(nunique < 2)',
                'n_unique': n_unique,
                'action': 'keep_original_or_drop'
            }

        # 第三步：检查样本数量是否足够
        min_samples = algo_req.get('min_samples', 2)
        if n_samples < min_samples:
            return {
                'can_standardize': False,
                'reason': f'样本数不足，需要至少{min_samples}个，当前有{n_samples}个',
                'n_samples': n_samples,
                'action': 'insufficient_data'
            }

        # 所有条件满足，可以标准化
        return {
            'can_standardize': True,
            'recommended_methods': algo_req['priority'],
            'n_unique': n_unique,
            'n_samples': n_samples,
            'action': 'proceed_standardization'
        }

    # 当样本数没有达到要求时，有几种处理策略，具体取决于算法和业务场景

    def complete_feature_processing(series: pd.Series, algorithm: str) -> Dict[str, Any]:
        """完整的特征处理流程"""
        series_clean = series.dropna()
        n_samples = len(series_clean)
        n_unique = series_clean.nunique()

        # 获取算法要求
        algo_req = self.algorithm_requirements[algorithm]
        required_samples = algo_req.get('min_samples', 2)

        # 检查标准化可行性
        standardization_check = can_standardize_feature(series, algorithm)

        result = {
            'algorithm': algorithm,
            'n_samples': n_samples,
            'n_unique': n_unique,
            'required_samples': required_samples,
            'standardization_recommended': algo_req['standardization'] in ['required', 'recommended']
        }

        if standardization_check['can_standardize']:
            # 正常标准化流程
            result.update(standardization_check)
            result['final_series'] = apply_standardization(series, algo_req['priority'][0])

        else:
            # 样本不足的处理
            if '样本数不足' in standardization_check['reason']:
                strategy_info = handle_insufficient_samples(series, algorithm, n_samples, required_samples)
                processing_result = apply_insufficient_samples_strategy(series, strategy_info['strategy'], algorithm)

                result.update({
                    'standardization_applied': processing_result['standardization_applied'],
                    'handling_strategy': strategy_info['strategy'],
                    'final_series': processing_result['processed_series'],
                    'warning': processing_result['warning'],
                    'action_taken': processing_result['action_taken']
                })
            else:
                # 其他原因（如常数列）
                result.update(standardization_check)
                result['final_series'] = series  # 保持原样

        return result

    def get_recommendation(processing_result: Dict) -> str:
        """根据处理结果给出建议"""
        if processing_result['standardization_applied']:
            return "✅ 特征已成功标准化"

        elif processing_result['handling_strategy'] == 'keep_original':
            return "⚠️  特征保持原样（算法能处理）"

        elif processing_result['handling_strategy'] == 'keep_with_warning':
            return "⚠️  特征保持原样（结果需谨慎解读）"

        elif processing_result['handling_strategy'] == 'drop_or_impute':
            if processing_result['final_series'] is None:
                return "❌ 建议删除该特征"
            else:
                return "⚠️  已尝试填充处理，建议验证效果"

        else:
            return "ℹ️  按默认策略处理"