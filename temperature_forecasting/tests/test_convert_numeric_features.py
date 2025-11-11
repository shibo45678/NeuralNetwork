import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from unittest.mock import patch
import warnings
from data.data_preparation.convert_numeric_features import ConvertNumericColumns


# 测试类
class TestConvertNumericColumns:
    """ConvertNumericColumns 转换器的综合测试套件"""

    def setup_method(self):
        """测试初始化"""
        self.sample_data = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_numeric_col': ['1', '2', '3', '4', '5'],
            'str_mixed_col1': ['1', '2.5', 'abc', '4', '5.5'],
            'str_mixed_col2': ['1', '2', 'abc', '4', '5'],
            'exclude_col': [10, 20, 30, 40, 50],  # 用于排除测试
            'bool_col': [True, False, True, False, True]
        })

    # ===== 单元测试 =====

    def test_initialization_default_params(self):
        """测试默认参数初始化"""
        transformer = ConvertNumericColumns()
        assert transformer.numeric_columns == []
        assert transformer.preserve_object_integer_types is True
        assert transformer.exclude_cols == []
        assert transformer.pass_through is False

    def test_initialization_custom_params(self):
        """测试自定义参数初始化"""
        transformer = ConvertNumericColumns(
            cols=['col1', 'col2'],
            preserve_object_integer_types=False,
            exclude_cols=['time_col'],
            pass_through=True
        )
        assert transformer.numeric_columns == ['col1', 'col2']
        assert transformer.preserve_object_integer_types is False
        assert transformer.exclude_cols == ['time_col']
        assert transformer.pass_through is True

    def test_fit_with_specified_columns(self):
        """测试指定列名的拟合过程"""
        transformer = ConvertNumericColumns(cols=['int_col', 'float_col'])
        transformer.fit(self.sample_data)

        assert 'int_col' in transformer.numeric_columns
        assert 'float_col' in transformer.numeric_columns
        assert 'int_col' in transformer.original_dtypes_
        assert 'float_col' in transformer.original_dtypes_

    def test_fit_with_none_columns_auto_detection(self):
        """测试自动检测数值列"""
        transformer = ConvertNumericColumns(cols=None, exclude_cols=['exclude_col'])
        transformer.fit(self.sample_data)

        expected_cols = ['int_col', 'float_col']
        for col in expected_cols:
            assert col in transformer.numeric_columns
        assert 'exclude_col' not in transformer.numeric_columns

    def test_fit_with_missing_columns_warning(self):
        """测试处理不存在的列时发出警告"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            transformer = ConvertNumericColumns(cols=['int_col', 'nonexistent_col'])
            transformer.fit(self.sample_data)

            assert len(w) == 1
            assert "警告: 以下指定列不存在" in str(w[0].message)
            assert 'nonexistent_col' in str(w[0].message)

    def test_transform_basic_conversion(self):
        """测试基本类型转换功能"""
        transformer = ConvertNumericColumns(cols=['str_numeric_col', 'str_mixed_col1'])
        transformer.fit(self.sample_data)

        result = transformer.transform(self.sample_data)

        # 检查数值字符串列正确转换
        # assert result['str_numeric_col'].dtype in [np.dtype('float64'), np.dtype('Int64')] 错 大写才是pandas里面的
        dtype_str = str(result['str_numeric_col'].dtype)
        assert dtype_str in ['Int64']  # pandas
        assert result['str_numeric_col'].tolist() == [1, 2, 3, 4, 5]

        # 检查混合列正确处理（非数值转为NaN）
        expected_mixed = [1.0, 2.5, np.nan, 4.0, 5.5]
        for i, (actual, expected) in enumerate(zip(result['str_mixed_col1'], expected_mixed)):
            if np.isnan(expected):
                assert np.isnan(actual), f"Index {i}: expected NaN, got {actual}"
            else:
                assert actual == expected, f"Index {i}: expected {expected}, got {actual}"

    def test_transform_preserve_integer_types(self):
        """测试(object对象里面的整数类型）保持功能"""
        # 测试保持整数类型
        transformer_preserve = ConvertNumericColumns(
            cols=['str_mixed_col2'],
            preserve_object_integer_types=True
        )
        transformer_preserve.fit(self.sample_data)
        result_preserve = transformer_preserve.transform(self.sample_data)

        # 测试不保持整数类型
        transformer_no_preserve = ConvertNumericColumns(
            cols=['str_mixed_col2'],
            preserve_object_integer_types=False
        )
        transformer_no_preserve.fit(self.sample_data)
        result_no_preserve = transformer_no_preserve.transform(self.sample_data)

        # 验证类型差异
        assert str(result_preserve['str_mixed_col2'].dtype) == 'Int64'
        assert result_no_preserve['str_mixed_col2'].dtype == np.dtype('float64')

    def test_transform_exclude_columns(self):
        """测试排除列功能"""
        transformer = ConvertNumericColumns(
            cols=None,
            exclude_cols=['exclude_col', 'bool_col']
        )
        transformer.fit(self.sample_data)
        result = transformer.transform(self.sample_data)

        # 检查排除列未被处理
        assert 'exclude_col' not in transformer.numeric_columns
        assert result['exclude_col'].dtype == np.int64  # 保持原类型

    def test_transform_pass_through_mode(self):
        """测试直通模式"""
        transformer = ConvertNumericColumns(pass_through=True)
        transformer.fit(self.sample_data)
        result = transformer.transform(self.sample_data)

        # 在直通模式下，数据应原样返回
        pd.testing.assert_frame_equal(result, self.sample_data)

    def test_transform_with_numeric_columns_empty(self):
        """测试无数值列需要处理的情况"""
        transformer = ConvertNumericColumns(cols=[])
        transformer.fit(self.sample_data)
        result = transformer.transform(self.sample_data)

        pd.testing.assert_frame_equal(result, self.sample_data)

    def test_fit_transform_integration(self):
        """测试拟合和转换的集成"""
        transformer = ConvertNumericColumns(cols=['str_mixed_col1'])
        result = transformer.fit_transform(self.sample_data)

        assert 'str_mixed_col1' in result.columns
        assert pd.api.types.is_numeric_dtype(result['str_mixed_col1'])

    # ===== 边界值测试 =====

    def test_empty_dataframe(self):
        """测试空DataFrame处理"""
        empty_df = pd.DataFrame()
        transformer = ConvertNumericColumns(cols=[])

        with pytest.raises(ValueError):
            result = transformer.fit_transform(empty_df)

    def test_all_nan_column(self):
        """测试全NaN列的处理"""
        nan_data = pd.DataFrame({
            'all_nan_col': [np.nan, np.nan, np.nan],
            'mixed_nan_col': ['1', np.nan, '3']
        })

        transformer = ConvertNumericColumns(cols=['all_nan_col', 'mixed_nan_col'])
        result = transformer.fit_transform(nan_data)

        assert pd.api.types.is_float_dtype(result['all_nan_col'])
        assert pd.api.types.is_numeric_dtype(result['mixed_nan_col'])

    def test_large_numbers_conversion(self):
        """测试大数值转换"""
        large_data = pd.DataFrame({
            'large_int': ['999999999999999', '1000000000000000'],
            'scientific_notation': ['1.5e10', '2.3e-5']
        })

        transformer = ConvertNumericColumns(cols=['large_int', 'scientific_notation'])
        result = transformer.fit_transform(large_data)

        assert result['large_int'].iloc[0] == 999999999999999
        assert abs(result['scientific_notation'].iloc[0] - 1.5e10) < 1e-5

    # ===== 异常情况测试 =====

    def test_invalid_input_type(self):
        """测试无效输入类型"""
        transformer = ConvertNumericColumns()

        with pytest.raises(Exception):  # 具体异常类型取决于validate_input装饰器
            transformer.fit("invalid_input")

    def test_column_with_special_characters(self):
        """测试特殊字符列名"""
        special_data = pd.DataFrame({
            'col-with-dash': ['1', '2', '3'],
            'col with space': ['4', '5', '6'],
            'col.dotted': ['7', '8', '9']
        })

        transformer = ConvertNumericColumns(cols=['col-with-dash', 'col with space', 'col.dotted'])
        result = transformer.fit_transform(special_data)

        for col in ['col-with-dash', 'col with space', 'col.dotted']:
            assert pd.api.types.is_numeric_dtype(result[col])

    # ===== 集成测试 =====

    def test_in_sklearn_pipeline(self):
        """测试在sklearn Pipeline中的集成"""
        # 创建测试数据
        X = pd.DataFrame({
            'feature1': ['1', '2', '3', '4', '5'],
            'feature2': ['1.1', '2.2', '3.3', '4.4', '5.5'],
            'target': [0, 1, 0, 1, 0]
        })
        y = X.pop('target')

        # 创建管道
        pipeline = Pipeline([
            ('numeric_converter', ConvertNumericColumns(cols=['feature1', 'feature2'])),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])

        # 测试管道运行
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})

    def test_data_consistency_through_pipeline(self):
        """测试管道中的数据一致性"""
        original_data = self.sample_data.copy()

        pipeline = Pipeline([
            ('converter', ConvertNumericColumns(
                cols=['str_numeric_col', 'str_mixed_col1'],
                exclude_cols=['exclude_col']
            ))
        ])

        transformed_data = pipeline.fit_transform(original_data)

        # 检查未指定列保持不变
        assert original_data['int_col'].equals(transformed_data['int_col'])
        assert original_data['exclude_col'].equals(transformed_data['exclude_col'])

        # 检查指定列已转换
        assert pd.api.types.is_numeric_dtype(transformed_data['str_numeric_col'])
        assert pd.api.types.is_numeric_dtype(transformed_data['str_mixed_col1'])

    # ===== 功能测试 =====

    def test_business_requirement_numeric_conversion(self):
        """测试业务需求：确保所有指定列正确转换为数值类型"""
        test_data = pd.DataFrame({
            'sales': ['1000', '2000', '3000'],
            'profit': ['500.50', '750.25', '1200.75'],
            'region': ['North', 'South', 'East']  # 非数值列
        })

        transformer = ConvertNumericColumns(cols=['sales', 'profit'])
        result = transformer.fit_transform(test_data)

        # 业务需求验证
        assert pd.api.types.is_numeric_dtype(result['sales']), "销售额必须为数值类型"
        assert pd.api.types.is_numeric_dtype(result['profit']), "利润必须为数值类型"
        assert result['sales'].sum() == 6000, "销售额总和计算正确"
        assert abs(result['profit'].sum() - 2451.5) < 0.01, "利润总和计算正确"

    def test_data_quality_after_conversion(self):
        """测试转换后的数据质量"""
        # 创建包含各种数据质量问题的测试数据
        quality_data = pd.DataFrame({
            'valid_ints': ['1', '2', '3', '4', '5'],
            'valid_floats': ['1.1', '2.2', '3.3', '4.4', '5.5'],
            'with_invalid': ['1', 'invalid', '3', '4.5.6', '7'],
            'with_nulls': ['1', None, '3', np.nan, '5']
        })

        transformer = ConvertNumericColumns(cols=quality_data.columns.tolist())
        result = transformer.fit_transform(quality_data)

        # 数据质量检查
        for col in quality_data.columns:
            # 检查列是否为数值类型
            assert pd.api.types.is_numeric_dtype(result[col]), f"列 {col} 应为数值类型"

            # 检查没有因为转换而丢失有效数据
            original_valid = pd.to_numeric(quality_data[col], errors='coerce').notna().sum()
            transformed_valid = result[col].notna().sum()
            assert original_valid == transformed_valid, f"列 {col} 有效数据数量应保持不变"

    def test_performance_large_dataset(self):
        """测试大数据集性能"""
        # 创建大型测试数据集
        n_rows = 10000
        large_data = pd.DataFrame({
            f'col_{i}': [str(x) for x in range(n_rows)]
            for i in range(5)
        })

        transformer = ConvertNumericColumns(cols=large_data.columns.tolist())

        # 性能测试：确保在合理时间内完成
        import time
        start_time = time.time()

        result = transformer.fit_transform(large_data)

        end_time = time.time()
        processing_time = end_time - start_time

        # 验证处理时间在可接受范围内（可根据实际情况调整）
        assert processing_time < 10.0, f"处理时间过长: {processing_time:.2f}秒"

        # 验证所有列正确转换
        for col in large_data.columns:
            assert pd.api.types.is_numeric_dtype(result[col])

    def test_user_scenario_ecommerce_data(self):
        """测试用户场景：电商数据处理"""
        # 模拟电商数据场景
        ecommerce_data = pd.DataFrame({
            'order_id': ['1001', '1002', '1003', '1004'],
            'product_price': ['29.99', '49.99', '19.99', '99.99'],
            'quantity': ['2', '1', '3', '1'],
            'discount_percentage': ['10', '0', '15', '5'],
            'customer_rating': ['4.5', '3.5', '5.0', '4.0'],
            'product_name': ['Widget A', 'Widget B', 'Widget C', 'Widget D']  # 非数值
        })

        numeric_cols = ['product_price', 'quantity', 'discount_percentage', 'customer_rating']

        transformer = ConvertNumericColumns(cols=numeric_cols)
        result = transformer.fit_transform(ecommerce_data)

        # 验证业务计算
        result['total_price'] = result['product_price'] * result['quantity']
        result['final_price'] = result['total_price'] * (1 - result['discount_percentage'] / 100)

        # 业务逻辑验证
        assert result['final_price'].notna().all(), "所有最终价格应有效计算"
        assert (result['final_price'] >= 0).all(), "最终价格应为非负数"
        assert (result['customer_rating'] >= 0).all() and (result['customer_rating'] <= 5).all(), "评分应在0-5范围内"

    # ===== 装饰器功能测试 =====

    def test_validate_input_decorator_compatibility(self):
        """测试与输入验证装饰器的兼容性"""
        # 这个测试验证装饰器不会干扰正常功能
        transformer = ConvertNumericColumns(cols=['int_col', 'float_col'])

        # 应该正常执行，不抛出异常
        result = transformer.fit_transform(self.sample_data)
        assert isinstance(result, pd.DataFrame)

    def test_error_handling_with_validate_input(self):
        """测试输入验证装饰器的错误处理"""
        dict = {'a': [1, 2, 3],
                'b': [3, 4, 6]
                }
        transformer = ConvertNumericColumns()

        # 测试无效输入（取决于validate_input的具体实现）
        # 这里我们假设装饰器会验证输入类型
        with pytest.raises((TypeError,ValueError)):
            transformer.fit(dict)  # 无效的列表输入



if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
