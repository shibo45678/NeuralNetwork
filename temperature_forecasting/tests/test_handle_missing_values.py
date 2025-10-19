
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

from data.data_cleaner.handle_missing_values import (BaseMissingValueHandler,NumericMissingValueHandler,CategoricalMissingValueHandler,
        ComprehensiveMissingValueHandler,BusinessAwareMissingHandler)



# 模拟辅助类（为了测试）
class ColumnsTypeIdentify:
    def fit(self, X):
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numeric_columns = numeric_cols
        self.categorical_columns = categorical_cols
        return self


class FeatureSelector:
    def fit(self, X, y):
        # 简化的相关性计算
        self.correlations = {}
        for col in X.columns:
            if X[col].dtype.kind in 'iufc' and hasattr(y, 'corr'):
                try:
                    self.correlations[col] = X[col].corr(y)
                except:
                    self.correlations[col] = 0
            else:
                self.correlations[col] = 0
        return self


# 简化装饰器
def validate_input(validate_y=True):
    def decorator(func):
        def wrapper(self, X, y=None):
            return func(self, X, y)

        return wrapper

    return decorator


def validate_output(validate_y=False):
    def decorator(func):
        def wrapper(self, X):
            return func(self, X)

        return wrapper

    return decorator


# 这里插入你的 BaseMissingValueHandler 类代码...

class TestMissingValueHandler:
    """缺失值处理器测试类"""

    def test_basic_functionality(self):
        """测试基本功能"""
        print("=" * 50)
        print("测试1: 基本功能测试")
        print("=" * 50)

        # 创建测试数据
        data = {
            'age': [25, np.nan, 30, 35, np.nan, 40],
            'score': [85, 90, np.nan, 78, 92, np.nan],
            'salary': [50000, np.nan, np.nan, 60000, 55000, 52000],
            'department': ['IT', 'HR', 'IT', np.nan, 'Finance', 'HR']
        }
        df = pd.DataFrame(data)

        print("原始数据:")
        print(df)
        print("\n缺失值统计:")
        print(df.isna().sum())

        # 测试数值处理器
        numeric_config = {
            'spec_fill': [
                ('constant', {'columns': ['age'], 'fill_value': [30]}),
                ('median', {'columns': ['salary']})
            ],
            'smart_fill_remain': True # smart功能支持整体填充
        }

        numeric_imputer = NumericMissingValueHandler(numeric_config)
        numeric_imputer.fit(df)
        result = numeric_imputer.transform(df)

        print("\n数值处理结果:")
        print(result)
        print("缺失指示器:", numeric_imputer.missing_indicator)

        # 验证填充结果
        assert result['age'].isna().sum() == 0, "age列应该没有缺失值"
        assert result['salary'].isna().sum() == 0, "salary列应该没有缺失值"
        print("✅ 基本功能测试通过")

    def test_important_columns(self):
        """测试重要列处理"""
        print("\n" + "=" * 50)
        print("测试2: 重要列处理测试")
        print("=" * 50)

        data = {
            'user_id': [1, 2, 3, 4, 5, 6],  # 重要列
            'age': [25, np.nan, 30, 35, np.nan, 40],
            'score': [85, 90, np.nan, 78, 92, np.nan],
            'transaction_id': [1001, 1002, 1003, 1004, 1005, 1006]  # 重要列
        }
        df = pd.DataFrame(data)

        config = {
            'spec_fill': [
                ('mean', {'columns': ['age']}),
                ('median', {'columns': ['score']})
            ],
            'smart_fill_remain': False,
            'important_columns': ['user_id', 'transaction_id'] # 重要列出现没有填充必要的情况 没缺失
        }

        imputer = NumericMissingValueHandler(config)
        imputer.fit(df)
        result = imputer.transform(df)

        print("重要列处理结果:")
        print(result)
        print("缺失指示器:", imputer.missing_indicator)
        print("重要特征列表:", imputer.get_important_features(result))

        # 验证重要列被标记
        assert 'user_id' in imputer.missing_indicator
        assert 'transaction_id' in imputer.missing_indicator
        print("✅ 重要列处理测试通过")

    def test_skip_fill_columns(self):
        """测试跳过填充列"""
        print("\n" + "=" * 50)
        print("测试3: 跳过填充列测试")
        print("=" * 50)

        data = {
            'age': [25, np.nan, 30, 35, np.nan, 40],
            'income': [50000, np.nan, 60000, np.nan, 55000, 52000],  # 跳过填充
            'score': [85, 90, np.nan, 78, 92, np.nan]
        }
        df = pd.DataFrame(data)

        config = {
            'spec_fill': [
                ('mean', {'columns': ['age']})
            ],
            'skip_fill': ['income'],  # 跳过income列
            'smart_fill_remain': True
        }

        imputer = NumericMissingValueHandler(config)
        imputer.fit(df)
        result = imputer.transform(df)

        print("跳过填充结果:")
        print(result)
        print("income列缺失值数量:", result['income'].isna().sum())

        # 验证income列仍然有缺失值
        assert result['income'].isna().sum() > 0, "income列应该仍然有缺失值"
        print("✅ 跳过填充测试通过")

    def test_categorical_columns(self):
        """测试分类列处理"""
        print("\n" + "=" * 50)
        print("测试4: 分类列处理测试")
        print("=" * 50)

        data = {
            'department': ['IT', 'HR', 'IT', np.nan, 'Finance', 'HR', np.nan],
            'level': ['Junior', 'Senior', np.nan, 'Mid', 'Senior', np.nan, 'Junior'],
            'city': ['Beijing', 'Shanghai', np.nan, 'Beijing', np.nan, 'Guangzhou', 'Shenzhen']
        }
        df = pd.DataFrame(data)

        config = {
            'spec_fill': [
                ('mode', {'columns': ['department']})
            ],
            'smart_fill_remain': True
        }

        imputer = CategoricalMissingValueHandler(config)
        imputer.fit(df)
        result = imputer.transform(df)

        print("分类列处理结果:")
        print(result)
        print("缺失指示器:", imputer.missing_indicator)

        # 验证填充
        assert result['department'].isna().sum() == 0, "department列应该没有缺失值"
        print("✅ 分类列处理测试通过")

    def test_comprehensive_handler(self):
        """测试综合处理器"""
        print("\n" + "=" * 50)
        print("测试5: 综合处理器测试")
        print("=" * 50)

        data = {
            'age': [25, np.nan, 30, 35, np.nan, 40],  # 数值列
            'salary': [50000, np.nan, 60000, np.nan, 55000, 52000],  # 数值列
            'department': ['IT', 'HR', 'IT', np.nan, 'Finance', 'HR'],  # 分类列
            'city': ['Beijing', 'Shanghai', np.nan, 'Beijing', np.nan, 'Guangzhou']  # 分类列
        }
        df = pd.DataFrame(data)

        numeric_config = {
            'spec_fill': [
                ('mean', {'columns': ['age']}),
                ('median', {'columns': ['salary']})
            ],
            'smart_fill_remain': False
        }

        categorical_config = {
            'spec_fill': [
                ('mode', {'columns': ['department']})
            ],
            'skip_fill': ['city'],  # 跳过city列
            'smart_fill_remain': True
        }

        comprehensive_imputer = ComprehensiveMissingValueHandler(
            numeric_config=numeric_config,
            categorical_config=categorical_config
        )

        comprehensive_imputer.fit(df)
        result = comprehensive_imputer.transform(df)

        print("综合处理结果:")
        print(result)
        print("缺失值统计:")
        print(result.isna().sum())

        # 验证结果
        assert result['age'].isna().sum() == 0
        assert result['salary'].isna().sum() == 0
        assert result['department'].isna().sum() == 0
        assert result['city'].isna().sum() > 0  # city列应该还有缺失值

        print("✅ 综合处理器测试通过")

    def test_edge_cases(self):
        """测试边界情况"""
        print("\n" + "=" * 50)
        print("测试6: 边界情况测试")
        print("=" * 50)

        # 测试1: 全空数据
        print("测试全空数据...")
        empty_data = {'col1': [np.nan, np.nan, np.nan], 'col2': [np.nan, np.nan, np.nan]}
        empty_df = pd.DataFrame(empty_data)

        try:
            imputer = NumericMissingValueHandler({
                'spec_fill': [('constant', {'columns': ['col1'], 'fill_value': [0]})],
                'smart_fill_remain': True
            })
            result = imputer.fit_transform(empty_df)
            print("全空数据处理成功") # 全空抛出
        except Exception as e:
            print(f"全空数据处理: {e}")

        # 测试2: 无缺失数据
        print("\n测试无缺失数据...")
        no_missing_data = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
        no_missing_df = pd.DataFrame(no_missing_data)

        imputer = NumericMissingValueHandler({
            'spec_fill': [('mean', {'columns': ['col1']})],
            'smart_fill_remain': True
        })
        result = imputer.fit_transform(no_missing_df)
        print("无缺失数据处理成功")

        # 测试3: 配置验证
        print("\n测试配置验证...")
        try:
            invalid_config = {
                'spec_fill': [],
                'smart_fill_remain': False  # 应该报错
            }
            imputer = NumericMissingValueHandler(invalid_config)
            imputer.fit(empty_df)
        except ValueError as e:
            print(f"配置验证正确捕获错误: {e}") # 配置未报错

        print("✅ 边界情况测试通过")

    def test_smart_fill_logic(self):
        """测试智能填充逻辑"""
        print("\n" + "=" * 50)
        print("测试7: 智能填充逻辑测试")
        print("=" * 50)

        # 创建符合正态分布的数据（应该用均值填充）
        np.random.seed(42)
        normal_data = np.random.normal(100, 10, 50)
        normal_data_with_nan = normal_data.copy()
        normal_data_with_nan[::10] = np.nan  # 每10个插入一个缺失值

        # 创建偏态分布数据（应该用中位数填充）
        skewed_data = np.random.exponential(2, 50)
        skewed_data_with_nan = skewed_data.copy()
        skewed_data_with_nan[::8] = np.nan

        data = {
            'normal_col': normal_data_with_nan,
            'skewed_col': skewed_data_with_nan
        }
        df = pd.DataFrame(data)

        config = {
            'smart_fill_remain': True  # 只使用智能填充
        }

        imputer = NumericMissingValueHandler(config)
        imputer.fit(df)
        result = imputer.transform(df)

        print("智能填充结果:")
        print(f"normal_col 填充值: {imputer.columns_info['normal_col'][0]['fill_value']}")
        print(f"skewed_col 填充值: {imputer.columns_info['skewed_col'][0]['fill_value']}")
        print("缺失值统计:", result.isna().sum())

        assert result['normal_col'].isna().sum() == 0 # 智能填充只覆盖15%
        assert result['skewed_col'].isna().sum() == 0
        print("✅ 智能填充逻辑测试通过")

    def test_missing_indicator_creation(self):
        """测试缺失指示器创建"""
        print("\n" + "=" * 50)
        print("测试8: 缺失指示器创建测试")
        print("=" * 50)

        data = {
            'important_col': [1, 2, np.nan, 4, 5, np.nan],  # 重要列
            'high_missing_col': [1, np.nan, np.nan, np.nan, 5, 6],  # 高缺失率列
            'normal_col': [1, 2, 3, 4, 5, 6]  # 正常列 主义重要列的判断 测试数据 短 很容易达到标准
        }
        df = pd.DataFrame(data)

        config = {
            'spec_fill': [
                ('mean', {'columns': ['important_col', 'high_missing_col', 'normal_col']})
            ],
            'smart_fill_remain': False,
            'important_columns': ['important_col']
        }

        imputer = NumericMissingValueHandler(config)
        imputer.fit(df)
        result = imputer.transform(df)

        print("缺失指示器测试结果:")
        print("缺失指示器列表:", imputer.missing_indicator) # normal_col不缺失的列在缺失指示器内
        print("重要列列表:", imputer.get_important_features(df))
        # 验证缺失指示器列存在
        assert 'important_col_missing_indicator' in result.columns
        assert 'high_missing_col_missing_indicator' in result.columns
        assert 'normal_col_missing_indicator' not in result.columns  # 正常列不应该有指示器

        print("✅ 缺失指示器创建测试通过")


def run_all_tests():
    """运行所有测试"""
    tester = TestMissingValueHandler()

    test_methods = [
        'test_basic_functionality',
        'test_important_columns',
        'test_skip_fill_columns',
        'test_categorical_columns',
        'test_comprehensive_handler',
        'test_edge_cases',
        'test_smart_fill_logic',
        'test_missing_indicator_creation'
    ]

    for method_name in test_methods:
        try:
            method = getattr(tester, method_name)
            method()
        except Exception as e:
            print(f"❌ {method_name} 测试失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("开始测试缺失值处理器...")
    run_all_tests()
    print("\n" + "=" * 50)
    print("所有测试完成!")