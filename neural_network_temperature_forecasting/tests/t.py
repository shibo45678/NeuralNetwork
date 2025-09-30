# 测试代码
import pandas as pd
import numpy as np
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
from src.data.processing import DataLoader ,DescribeData,ProblemColumnsFixed,SpecialColumnsFixed,ColumnsTypeIdentify,ProcessNumericColumns

# 创建测试数据
test_data = pd.DataFrame({
    'age': ['25', '30', '35', 'abc'],  # 有字符串需要转换
    'salary': [50000, 60000, 70000, 80000],
    'height': [175.5, 180.2, 165.3, 170.1],
    'name': ['Alice', 'Bob', 'Charlie', 'David']  # 非数值列
})

print("原始数据:")
print(test_data)
print(test_data.dtypes)

# 测试1: 不指定列（自动识别）
processor1 = ProcessNumericColumns(preserve_integer_types=False)
result1 = processor1.fit_transform(test_data)
print("\n自动识别结果:")
print(result1.dtypes)

# 测试2: 指定特定列
processor2 = ProcessNumericColumns(cols=['age', 'salary'],preserve_integer_types=True)
result2 = processor2.fit_transform(test_data)
print("\n指定列结果:")
print(result2.dtypes)

# 测试3: 指定不存在的列
processor3 = ProcessNumericColumns(cols=['age', 'nonexistent_col'],preserve_integer_types=True)
result3 = processor3.fit_transform(test_data)