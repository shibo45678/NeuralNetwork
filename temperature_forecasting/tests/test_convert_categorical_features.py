# 测试代码
import pandas as pd
import numpy as np
from data.data_preparation.convert_categorical_features import ConvertCategoricalColumns
from sklearn.pipeline import Pipeline


def test_process_categorical_columns():
    """测试 ConvertCategoricalColumns 类"""
    print("=" * 50)
    print("开始测试 ConvertCategoricalColumns")
    print("=" * 50)

    # 创建测试数据
    np.random.seed(42)
    n_samples = 20

    test_data = pd.DataFrame({
        'Date Time': [
            '01.01.2023 08:00:00', '01.01.2023 09:00:00', '01.01.2023 10:00:00',
            '02.01.2023 08:00:00', '02.01.2023 09:00:00', '02.01.2023 10:00:00',
            '03.01.2023 08:00:00', '03.01.2023 09:00:00', '03.01.2023 10:00:00',
            '01.01.2023 11:00:00', '01.01.2023 12:00:00', '01.01.2023 13:00:00',
            '02.01.2023 11:00:00', '02.01.2023 12:00:00', '02.01.2023 13:00:00',
            '03.01.2023 11:00:00', '03.01.2023 12:00:00', '03.01.2023 13:00:00',
            '04.01.2023 08:00:00', '04.01.2023 09:00:00'
        ],
        'season': np.random.choice(['spring', 'summer', 'autumn', 'winter'], n_samples),
        'weather': np.random.choice(['sunny', 'cloudy', 'rainy'], n_samples),
        'city': np.random.choice(['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen', 'Hangzhou'], n_samples),
        'temperature': np.random.uniform(15, 35, n_samples),
        'humidity': np.random.uniform(40, 90, n_samples)
    })

    print("原始数据:")
    print(test_data.head())
    print(f"原始数据形状: {test_data.shape}")
    print(f"原始数据列: {test_data.columns.tolist()}")
    print(f"原始数据类型:\n{test_data.dtypes}")
    print()

    # 测试场景1：自动检测分类列
    print("测试场景1: 自动检测分类列")
    print("-" * 30)
    processor1 = ConvertCategoricalColumns(categorical_columns=None)
    result1 = processor1.fit_transform(test_data)
    print(f"处理后的数据形状: {result1.shape}")
    print(f"处理后的数据列: {result1.columns.tolist()}")
    print(f"处理后的数据类型:\n{result1.dtypes}")
    print(f"时间列样例: {result1['Date Time'].head(3).tolist()}")
    print(f"输出res1结果：{result1.head(3)}")
    print()

    # 测试场景2：指定要处理的列
    print("测试场景2: 指定处理列")
    print("-" * 30)
    processor2 = ConvertCategoricalColumns(categorical_columns=['season', 'weather'])
    result2 = processor2.fit_transform(test_data)
    print(f"处理后的数据形状: {result2.shape}")
    print(f"处理后的数据列: {result2.columns.tolist()}")
    print(f"输出res2结果：{result2.head(3)}")
    print()

    # 测试场景3：在Pipeline中使用
    print("测试场景3: 在Pipeline中使用")
    print("-" * 30)
    pipeline = Pipeline([
        ('categorical_processor', ConvertCategoricalColumns(
            categorical_columns=['season', 'weather','city', 'Date Time']
        ))
    ])

    result3 = pipeline.fit_transform(test_data)
    print(f"Pipeline处理后的数据形状: {result3.shape}")
    print(f"Pipeline处理后的数据列: {result3.columns.tolist()}")
    print()

    # 验证独热编码结果
    print("验证独热编码结果:")
    print("-" * 30)
    season_columns = [col for col in result3.columns if col.startswith('season_')]
    weather_columns = [col for col in result3.columns if col.startswith('weather_')]

    print(f"季节独热编码列: {season_columns}")
    print(f"天气独热编码列: {weather_columns}")

    # 检查独热编码的值
    if season_columns:
        print("季节列独热编码样例:")
        print(result3[season_columns].head(3))

    if weather_columns:
        print("天气列独热编码样例:")
        print(result3[weather_columns].head(3))
    print()

    # 测试场景4：处理不存在的列
    print("测试场景4: 处理不存在的列")
    print("-" * 30)
    processor4 = ConvertCategoricalColumns(categorical_columns=['nonexistent_col', 'season'])
    result4 = processor4.fit_transform(test_data)
    print(f"处理后的数据形状: {result4.shape}")
    print(f"result3处理后的数据列: {result4.columns.tolist()}")
    print()




    # 总结
    print("=" * 50)
    print("测试总结:")
    print("=" * 50)
    print(f"✓ 时间列转换: {isinstance(result3['Date Time'].iloc[0], pd.Timestamp)}")
    print(f"✓ 数据排序: {result3['Date Time'].is_monotonic_increasing}") # Pandas Series 的属性，检查数据是否单调递增
    print(f"✓ 独热编码创建: {len(season_columns) > 0}")
    print(f"✓ 数值列保留: {'temperature' in result3.columns}")
    print(f"✓ 原始分类列移除: {'season' not in result3.columns}")
    print("所有测试完成!")


# 运行测试
if __name__ == "__main__":
    test_process_categorical_columns()