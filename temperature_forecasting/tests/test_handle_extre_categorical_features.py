from data.data_cleaner.handle_extre_categorical_features import CategoricalOutlierProcessor
import pandas as pd


def test_categorical_processor_focus():
    """重点测试 nan 拼写纠正和相似度计算"""

    print("=" * 70)
    print("重点测试：nan拼写纠正和相似度计算")
    print("=" * 70)

    # 1. 创建专门测试数据
    print("\n1. 创建专门测试数据...")

    # 定义每个列的测试数据
    nan_variations = [
                         'nan', 'NaN', 'NAN', 'Nan', 'NULL', 'null', 'Null', 'nUll',
                         'N/A', 'n/a', 'None', 'none', 'NONE', '', ' ', '  ',
                         '正常值1', '正常值2'
                     ] * 25  # 调整重复次数使长度合适

    similarity_test = [
                          'Shanghai', 'SHANGHAI', 'shanghai', 'ShangHai', 'Shangha', 'Shanghais',
                          'ShanghaA', 'Shanghais', 'Guangzhou', 'GUANGZHOU', 'guangzhou', 'Guanzhou',
                          'Guangzho', 'guangZHOU', 'Guangzho', 'guangzhOU', 'Beijing', 'BEIJING'
                      ] * 24

    boolean_test = [
                       'Yes', 'YES', 'yes', 'Y', 'y', 'True', 'TRUE', 'true',
                       'No', 'NO', 'no', 'N', 'n', 'False', 'FALSE', 'false',
                       'Maybe', 'Unknown'  # 添加一些其他值
                   ] * 27

    # 确保所有列表长度相同（取最小长度）
    min_length = min(len(nan_variations),len(similarity_test), len(boolean_test))

    test_data = {
        'nan_variations': nan_variations[:min_length],
        'similarity_test': similarity_test[:min_length],
        'boolean_test': boolean_test[:min_length]
    }

    df = pd.DataFrame(test_data)
    print(f"测试数据形状: {df.shape}")
    print(f"各列长度: nan_variations={len(test_data['nan_variations'])}, "
          f"similarity_test={len(test_data['similarity_test'])}, "
          f"boolean_test={len(test_data['boolean_test'])}")

    # 2. 初始化处理器
    print("\n2. 初始化处理器...")
    processor = CategoricalOutlierProcessor()

    # 3. 学习数据模式
    print("\n3. 学习数据模式...")
    processor.learn_categories(df)

    # 4. 测试 nan 拼写纠正
    print("\n4. 测试 nan 拼写纠正...")
    print("原始 nan 相关值分布:")
    nan_original = df['nan_variations'].value_counts()
    for val, count in nan_original.head(10).items():  # 只显示前10个
        print(f"  '{val}': {count}")

    # 应用清洗
    df_cleaned = processor.clean_categorical_data(df, strategy='consolidate')

    print("\n清洗后 nan 相关值分布:")
    nan_cleaned = df_cleaned['nan_variations'].value_counts(dropna=False)
    for val, count in nan_cleaned.head(10).items():
        if pd.isna(val):
            print(f"  NaN: {count}")
        else:
            print(f"  '{val}': {count}")

    # 检查 nan 标准化效果
    nan_variants = ['nan', 'NaN', 'NAN', 'Nan', 'NULL', 'null', 'Null', 'nUll', 'N/A', 'n/a', 'None', 'none', 'NONE', '', ' ', '  ',]
    nan_count_original = df['nan_variations'].isin(nan_variants).sum()
    nan_count_cleaned = df_cleaned['nan_variations'].isna().sum()

    print(f"\nNaN标准化效果:")
    print(f"  原始数据中各种nan变体数量: {nan_count_original}")
    print(f"  清洗后实际NaN数量: {nan_count_cleaned}")

    # 5. 测试相似度计算
    print("\n5. 测试相似度计算...")

    # 检查相似度检测结果
    for col in ['similarity_test', 'boolean_test']:
        if col in processor.category_stats_:
            stats = processor.category_stats_[col]
            if 'potential_typos' in stats:
                print(f"\n{col} 列的拼写错误检测:")
                for typo in stats['potential_typos'][:5]:  # 显示前5个
                    print(f"  '{typo['wrong']}' → '{typo['correct']}' (相似度: {typo['similarity']:.3f})")

                if len(stats['potential_typos']) > 5:
                    print(f"  ... 还有 {len(stats['potential_typos']) - 5} 个拼写错误")
            else:
                print(f"\n{col} 列未检测到拼写错误")

    # 6. 测试相似度计算的具体案例
    print("\n6. 相似度计算具体案例测试...")

    test_pairs = [
        ('Shanghai', 'shanghai'),  # 大小写不同
        ('Shanghai', 'Shangha'),  # 少一个字符
        ('Shanghai', 'Shanghais'),  # 多一个字符
        ('Guangzhou', 'GUANGZHOU'),  # 大小写
        ('Guangzhou', 'Guanzhou'),  # 拼写错误
        ('Yes', 'yes'),  # 大小写
        ('True', 'true'),  # 大小写
        ('Yes', 'No'),  # 完全不同
        ('', 'test'),  # 空字符串
        ('Shanghai', 'Shanghai'),  # 完全相同
    ]

    print("字符串相似度计算测试:")
    for str1, str2 in test_pairs:
        similarity = processor._calculate_similarity(str1, str2)
        print(f"  '{str1}' vs '{str2}': {similarity:.3f}")

    # 7. 测试修正映射
    print("\n7. 修正映射检查...")
    for col, correction_map in processor.correction_mappings_.items(): # 两个不同列存了同样的结果
        print(f"\n{col} 列的修正映射 (共{len(correction_map)}个):")

        if col == 'nan_variations':# 显示 nan 相关的修正
            nan_corrections = {}
            for wrong, correct in correction_map.items():
                wrong_str = str(wrong).lower()
                if any(nan_var in wrong_str for nan_var in ['nan', 'NaN', 'NAN', 'Nan', 'NULL', 'null', 'Null', 'nUll',
                         'N/A', 'n/a', 'None', 'none', 'NONE', '', ' ', '  ']):
                    nan_corrections[wrong] = correct

            if nan_corrections:
                print("  NaN相关修正:")
                for wrong, correct in list(nan_corrections.items())[:5]:
                    print(f"    '{wrong}' → {correct}")

        # 显示其他修正（最多显示5个）
        else:
            other_corrections = {k: v for k, v in correction_map.items()}
            if other_corrections:
                print("  其他修正:")
                count = 0
                for wrong, correct in other_corrections.items():
                    if count < 5:
                        print(f"    '{wrong}' → '{correct}'")
                        count += 1
                if len(other_corrections) > 5:
                    print(f"    ... 还有 {len(other_corrections) - 5} 个其他修正")

    # 8. 验证清洗效果
    print("\n8. 清洗效果验证...")

    for col in df.columns:
        original_unique = df[col].nunique()
        cleaned_unique = df_cleaned[col].nunique()
        reduction = original_unique - cleaned_unique

        print(f"  {col}: {original_unique} → {cleaned_unique} 个唯一值 (减少 {reduction})")

        # 检查特定列的改进
        if col == 'boolean_test':
            bool_original = set(df[col].unique())
            bool_cleaned = set(df_cleaned[col].unique())
            print(f"    布尔值标准化: {len(bool_original)} 种格式 → {len(bool_cleaned)} 种格式")

    # 9. 获取详细报告
    print("\n9. 详细报告...")
    report = processor.get_category_report()

    for col, stats in report['category_stats'].items():
        print(f"\n{col}:")
        print(f"  - 唯一值: {stats['unique_categories']}")
        print(f"  - 罕见类别: {stats['rare_categories_count']}")
        print(f"  - 拼写错误: {stats['potential_typos_count']}")
        if stats['most_frequent_category']:
            print(f"  - 最常见: '{stats['most_frequent_category']}' ({stats['most_frequent_frequency']:.1%})")

    print("\n" + "=" * 70)
    print("测试完成! ✅")
    print("=" * 70)

    return processor, df_cleaned


# 运行重点测试
if __name__ == "__main__":
    processor, cleaned_df = test_categorical_processor_focus()

    # 额外验证：检查具体的 nan 值转换
    print("\n额外验证 - 具体值转换示例:")
    original_nan_samples = ['nan', 'NaN', 'NULL', 'N/A', '']
    for sample in original_nan_samples:
        if sample in processor.correction_mappings_.get('nan_variations', {}):
            corrected = processor.correction_mappings_['nan_variations'][sample]
            print(f"  '{sample}' → {corrected}")
        else:
            print(f"  '{sample}' 未找到修正映射")
