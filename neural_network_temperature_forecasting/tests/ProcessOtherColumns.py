from data.processing import ProcessOtherColumns



def test_process_other_columns():
    """全面测试 ProcessOtherColumns 类的各种边界情况"""

    print("=" * 60)
    print("开始全面测试 ProcessOtherColumns")
    print("=" * 60)

    # 测试1: 正常的风向风速数据
    print("\n1. 测试正常的风向风速数据")
    df_normal = pd.DataFrame({
        'wind_direction': [0, 90, 180, 270, 45, 135, 225, 315],  # 风向（度）
        'wind_speed': [5, 8, 12, 6, 10, 15, 7, 9],  # 平均风速
        'max_wind_speed': [7, 10, 15, 8, 12, 18, 9, 11],  # 最大风速
        'temperature': [20, 22, 18, 19, 21, 23, 17, 20]  # 其他数据
    })

    try:
        processor = ProcessOtherColumns(
            dir_cols=['wind_direction'],
            var_cols=['wind_speed', 'max_wind_speed']
        )
        processor.fit(df_normal)
        result_normal = processor.transform(df_normal)
        print("✅ 正常数据处理成功")
        print("新增列:", [col for col in result_normal.columns if col in ['Wx', 'Wy', 'max_Wx', 'max_Wy']])
        print("转换结果:")
        print(result_normal[['Wx', 'Wy', 'max_Wx', 'max_Wy']].head())
    except Exception as e:
        print(f"❌ 正常数据处理失败: {e}")

    # 测试2: 只有平均风速，没有最大风速
    print("\n2. 测试只有平均风速的数据")
    df_single_var = pd.DataFrame({
        'wd': [0, 45, 90, 135],
        'wv': [3, 5, 7, 4],
        'other_col': [1, 2, 3, 4]
    })

    try:
        processor = ProcessOtherColumns(
            dir_cols=['wd'],
            var_cols=['wv']  # 只提供平均风速
        )
        processor.fit(df_single_var)
        result_single = processor.transform(df_single_var)
        print("✅ 单风速数据处理成功")
        print("max_Wx 和 max_Wy 应该与 Wx, Wy 相同")
        print(result_single[['Wx', 'Wy', 'max_Wx', 'max_Wy']].head())
    except Exception as e:
        print(f"❌ 单风速数据处理失败: {e}")

    # 测试3: 不存在的列
    print("\n3. 测试不存在的列")
    df_missing_cols = pd.DataFrame({
        'existing_col': [1, 2, 3],
        'value': [10, 20, 30]
    })

    try:
        processor = ProcessOtherColumns(
            dir_cols=['nonexistent_dir'],
            var_cols=['nonexistent_var']
        )
        processor.fit(df_missing_cols)
        result_missing = processor.transform(df_missing_cols)
        print("✅ 不存在列处理成功（应该跳过处理）")
        print("结果应该与原始数据相同")
        print("列数:", len(result_missing.columns))
    except Exception as e:
        print(f"❌ 不存在列处理失败: {e}")

    # 测试4: 空参数
    print("\n4. 测试空参数")
    df_empty_params = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [4, 5, 6]
    })

    try:
        processor = ProcessOtherColumns(dir_cols=None, var_cols=None)
        processor.fit(df_empty_params)
        result_empty = processor.transform(df_empty_params)
        print("✅ 空参数处理成功（应该跳过处理）")
        print("结果列:", result_empty.columns.tolist())
    except Exception as e:
        print(f"❌ 空参数处理失败: {e}")

    # 测试5: 部分列存在
    print("\n5. 测试部分列存在")
    df_partial = pd.DataFrame({
        'direction': [30, 60, 90],  # 存在的方向列
        'speed': [5, 6, 7],  # 存在的速度列
        'other': [1, 2, 3]  # 其他列
    })

    try:
        processor = ProcessOtherColumns(
            dir_cols=['direction', 'missing_dir'],  # 一个存在，一个不存在
            var_cols=['speed', 'missing_var']  # 一个存在，一个不存在
        )
        processor.fit(df_partial)
        result_partial = processor.transform(df_partial)
        print("✅ 部分列存在处理成功")
        print("应该只处理存在的列")
        print("结果列:", [col for col in result_partial.columns if col in ['Wx', 'Wy', 'max_Wx', 'max_Wy']])
    except Exception as e:
        print(f"❌ 部分列存在处理失败: {e}")

    # 测试6: 边界角度值
    print("\n6. 测试边界角度值")
    df_boundary = pd.DataFrame({
        'wind_dir': [0, 90, 180, 270, 360],  # 边界角度
        'wind_vel': [10, 10, 10, 10, 10],  # 固定风速
        'max_vel': [12, 12, 12, 12, 12]  # 固定最大风速
    })

    try:
        processor = ProcessOtherColumns(
            dir_cols=['wind_dir'],
            var_cols=['wind_vel', 'max_vel']
        )
        processor.fit(df_boundary)
        result_boundary = processor.transform(df_boundary)
        print("✅ 边界角度处理成功")
        print("角度 0° 应该对应 (10, 0)")
        print("角度 90° 应该对应 (0, 10)")
        print("角度 180° 应该对应 (-10, 0)")
        print("角度 270° 应该对应 (0, -10)")
        print("角度 360° 应该对应 (10, 0)")
        print("实际结果:")
        print(result_boundary[['Wx', 'Wy']])
    except Exception as e:
        print(f"❌ 边界角度处理失败: {e}")

    # 测试7: 零风速
    print("\n7. 测试零风速")
    df_zero_wind = pd.DataFrame({
        'dir': [45, 135, 225, 315],
        'speed': [0, 0, 0, 0],  # 零风速
        'max_speed': [0, 0, 0, 0]  # 零最大风速
    })

    try:
        processor = ProcessOtherColumns(
            dir_cols=['dir'],
            var_cols=['speed', 'max_speed']
        )
        processor.fit(df_zero_wind)
        result_zero = processor.transform(df_zero_wind)
        print("✅ 零风速处理成功")
        print("所有风矢量分量应该为 0")
        print(result_zero[['Wx', 'Wy', 'max_Wx', 'max_Wy']])
    except Exception as e:
        print(f"❌ 零风速处理失败: {e}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


# 运行测试
if __name__ == "__main__":
    test_process_other_columns()