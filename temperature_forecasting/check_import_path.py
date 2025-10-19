# import os
#
# def check_all_paths():
#     base_path = '/Users/shibo/Python/NeuralNetwork/temperature_forecasting'
#
#     paths_to_check = [
#         os.path.join(base_path, 'data', 'feature_engineer', 'feature_selector.py'),
#         os.path.join(base_path, 'src', 'data', 'feature_engineer', 'feature_selector.py'),
#         os.path.join(base_path, 'feature_selector.py'),
#     ]
#
#     print("检查可能的路径:")
#     for path in paths_to_check:
#         exists = os.path.exists(path)
#         status = "✓ 存在" if exists else "✗ 不存在"
#         print(f"  {status}: {path}")
#
#         if exists:
#             # 显示文件前几行
#             try:
#                 with open(path, 'r') as f:
#                     first_line = f.readline().strip()
#                     print(f"    第一行: {first_line}")
#             except Exception as e:
#                 print(f"    读取错误: {e}")
#
# check_all_paths()
