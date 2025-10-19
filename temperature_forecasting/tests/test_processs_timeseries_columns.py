import pandas as pd
from data.processing import ProcessTimeseriesColumns, TimeTypeConverter
import numpy as np


def test_process_timeseries_columns():
    """全面测试 ProcessTimeseriesColumns 类的各种边界情况"""

    # print("=" * 60)
    # print("开始全面测试 ProcessTimeseriesColumns")
    # print("=" * 60)
    #
    # # 测试1: 混合数据类型（你提到的problematic_data）
    # print("\n1. 测试混合数据类型")
    # problematic_data = [
    #     '2023-01-01',  # 字符串
    #     1672531200,  # Unix时间戳（秒）
    #     'invalid_date',  # 无效字符串
    #     44927,  # Excel日期
    #     '2023-01-02 14:30:00',  # 带时间的字符串
    #     None  # 空值
    # ]
    #
    # df_mixed = pd.DataFrame({
    #     'mixed_time': problematic_data,
    #     'value': range(len(problematic_data))
    # })
    #
    # print("混合数据测试:")
    # print(df_mixed)
    # print("数据类型:", df_mixed['mixed_time'].dtype)
    #
    # try:
    #     processor = ProcessTimeseriesColumns(col='mixed_time',auto_detect_string_format=True)
    #     processor.fit(df_mixed)
    #     result_mixed = processor.transform(df_mixed)
    #     print("✅ 混合数据已处理")
    #     print("处理后的数据类型:", result_mixed['mixed_time'].dtype)
    #     print("处理后的数据:")
    #     print(result_mixed[['mixed_time', 'Day_sin', 'Day_cos']])
    # except Exception as e:
    #     print(f"❌ 混合数据处理失败: {e}")
    #
    # 测试2: 纯字符串日期（带格式）
    print("\n2. 测试纯字符串日期（带格式）")
    df_string = pd.DataFrame({
        'string_date': ['20230101', '20230102', '20230103', '20230104', '20230104'],
        'auto_time': ['2023-01-01', '2023-01-02', '2023-01-03', np.nan, np.nan],
        'infer_time': ['01/01/2023', '12/01/2023', '31/12/2023', '12/31/2023', '12/01/2023'],
        'strange_time': ['2023-01-01 23:23:12', '00012023', '31/12/2023', '12/31/2023', '2024-01-01 23:23:12'],
        'value': [1, 2, 3, 4, 5],
        'perfect_seconds': [1633046400, 1633046401, 1633046402, 1633046403, 1633046406],
        'mixed_time': [1633046400, 1633046400, 1633046400, 1633046403.412, 1633046403.412],
        'excel_time': [44927, 44928, 44929, 44929.1,44930.1],
        'excel_wrong_time': [44927, 44928, 44929, 4430, 44929]

    })

    try:
        processor = ProcessTimeseriesColumns(format=None, auto_detect_string_format=True)
        processor.fit(df_string)
        result_string = processor.transform(df_string)
        print("✅ 字符串日期处理成功")
        print("处理后的时间列:", result_string['excel_time'
        ])
        print("处理后的季节列:", result_string['season'].tolist())
    except Exception as e:
        print(f"❌ 字符串日期处理失败: {e}")
    #
    # # 测试3: 字符串日期但没有提供format
    # print("\n3. 测试字符串日期但没有提供format")
    # df_string_no_format = pd.DataFrame({
    #     'string_date': ['2023-01-01', '2023-01-02'],
    #     'value': [1, 2]
    # })
    #
    # try:
    #     processor = ProcessTimeseriesColumns(col='string_date')
    #     processor.fit(df_string_no_format)
    #     result_no_format = processor.transform(df_string_no_format)
    #     print("❌ 应该报错但没有报错")
    # except ValueError as e:
    #     print(f"✅ 正确捕获错误: {e}")
    #
    # # 测试4: Unix时间戳
    # print("\n4. 测试Unix时间戳")
    # df_timestamp = pd.DataFrame({
    #     'timestamp': [1672531200, 1672617600, 1672704000],  # 2023-01-01, 2023-01-02, 2023-01-03
    #     'value': [10, 20, 30]
    # })
    #
    # try:
    #     processor = ProcessTimeseriesColumns(col='timestamp')
    #     processor.fit(df_timestamp)
    #     result_timestamp = processor.transform(df_timestamp)
    #     print("✅ Unix时间戳处理成功")
    #     print("时间跨度特征:", result_timestamp[['days_since_start', 'years_since_start']].head())
    # except Exception as e:
    #     print(f"❌ Unix时间戳处理失败: {e}")
    #
    # # 测试5: Excel日期数字
    # print("\n5. 测试Excel日期数字")
    # df_excel = pd.DataFrame({
    #     'excel_date': [44927, 44928, 44929],  # 2023-01-01, 2023-01-02, 2023-01-03
    #     'value': [100, 200, 300]
    # })
    #
    # try:
    #     processor = ProcessTimeseriesColumns(col='excel_date')
    #     processor.fit(df_excel)
    #     result_excel = processor.transform(df_excel)
    #     print("✅ Excel日期处理成功")
    #     print("处理后的时间列:", result_excel['excel_date'].head())
    # except Exception as e:
    #     print(f"❌ Excel日期处理失败: {e}")
    #
    # # 测试6: 自动识别datetime列
    # print("\n6. 测试自动识别datetime列")
    # df_auto = pd.DataFrame({
    #     'auto_timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
    #     'other_col': [1, 2, 3]
    # })
    #
    # try:
    #     processor = ProcessTimeseriesColumns(col=None)  # 自动识别
    #     processor.fit(df_auto)
    #     result_auto = processor.transform(df_auto)
    #     print("✅ 自动识别处理成功")
    #     print("识别的时间列:", processor.time_column)
    # except Exception as e:
    #     print(f"❌ 自动识别处理失败: {e}")
    #
    # # 测试7: 不存在的列
    # print("\n7. 测试不存在的列")
    # df_nonexistent = pd.DataFrame({
    #     'existing_col': [1, 2, 3],
    #     'value': [10, 20, 30]
    # })
    #
    # try:
    #     processor = ProcessTimeseriesColumns(col='nonexistent_col')
    #     processor.fit(df_nonexistent)
    #     result_nonexistent = processor.transform(df_nonexistent)
    #     print("✅ 不存在的列处理成功（应该跳过处理）")
    # except Exception as e:
    #     print(f"❌ 不存在的列处理失败: {e}")
    #
    # # 测试8: 空DataFrame
    # print("\n8. 测试空DataFrame")
    # df_empty = pd.DataFrame(columns=['time_col', 'value'])
    #
    # try:
    #     processor = ProcessTimeseriesColumns(col='time_col')
    #     processor.fit(df_empty)
    #     result_empty = processor.transform(df_empty)
    #     print("✅ 空DataFrame处理成功")
    # except Exception as e:
    #     print(f"❌ 空DataFrame处理结果: {e}") # 抛出空数据集的错

    # 测试9: 季节 is night，time_of_day 划分正确性
    # print("\n9. 测试季节划分正确性")
    # df_seasons = pd.DataFrame({
    #     'date': pd.date_range('2023-01-01', periods=20, freq='6h'),
    #     'value': range(20)
    # })
    #
    # try:
    #     processor = ProcessTimeseriesColumns(col='date')
    #     processor.fit(df_seasons)
    #     result_seasons = processor.transform(df_seasons)
    #     print("✅ 季节划分测试")
    #     season_months = result_seasons[['date', 'season','is_night','timedelta','days_since_start','years_since_start']].copy()
    #     season_months['datetime'] = pd.to_datetime(season_months['date'], unit='s')
    #     print(season_months[['date','datetime', 'season','is_night','timedelta','days_since_start','years_since_start']].to_string(index=False))
    #
    #     # 检查季节划分是否正确
    #     # 当使用分类变量分组时，pandas 默认会显示所有可能的分类组合（即使某些分类在数据中不存在
    #     season_mapping = result_seasons.groupby('is_night',observed=True)['date'].count()
    #     print("各季节数据点数量:", season_mapping.to_dict())
    #
    # except Exception as e:
    #     print(f"❌ 季节划分测试失败: {e}")

    # # 测试10: 周期编码功能
    # print("\n10. 测试周期编码功能")
    # df_cyclic = pd.DataFrame({
    #     'time': pd.date_range('2023-01-01', periods=10, freq='6h'),  # 每6小时一个点
    #     'value': range(10)
    # })
    #
    # try:
    #     processor = ProcessTimeseriesColumns(col='time')
    #     processor.fit(df_cyclic)
    #     result_cyclic = processor.transform(df_cyclic)
    #     print("✅ 周期编码测试")
    #     print("Day_sin范围:", f"{result_cyclic['Day_sin'].min():.3f} 到 {result_cyclic['Day_sin'].max():.3f}")
    #     print("Day_cos范围:", f"{result_cyclic['Day_cos'].min():.3f} 到 {result_cyclic['Day_cos'].max():.3f}")
    #     print("数值在有效范围内:",
    #           (result_cyclic['Day_sin'].between(-1, 1).all() and
    #            result_cyclic['Day_cos'].between(-1, 1).all()))
    # except Exception as e:
    #     print(f"❌ 周期编码测试失败: {e}")
    #
    # print("\n" + "=" * 60)
    # print("测试完成")
    # print("=" * 60)


# 运行测试
if __name__ == "__main__":
    test_process_timeseries_columns()
