# import sys
# import os
#
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
# sys.path.insert(0, project_root)
#
# print("=== 诊断信息 ===")
# print(f"工作目录: {os.getcwd()}")
# print(f"项目根目录: {project_root}")
#
# # 检查目录结构
# data_path = os.path.join(project_root, 'data')
# if os.path.exists(data_path):
#     print(f"✓ data 目录存在: {data_path}")
#     print(f"  data 目录内容: {os.listdir(data_path)}")
#
#     feature_engineer_path = os.path.join(data_path, 'feature_engineer')
#     if os.path.exists(feature_engineer_path):
#         print(f"✓ feature_engineer 目录存在: {feature_engineer_path}")
#         print(f"  feature_engineer 目录内容: {os.listdir(feature_engineer_path)}")
#
#         feature_selector_path = os.path.join(feature_engineer_path, 'feature_selector.py')
#         if os.path.exists(feature_selector_path):
#             print(f"✓ feature_selector.py 文件存在: {feature_selector_path}")
#         else:
#             print(f"✗ feature_selector.py 文件不存在")
#     else:
#         print(f"✗ feature_engineer 目录不存在")
# else:
#     print(f"✗ data 目录不存在")
#
# print(f"\nPython 路径:")
# for path in sys.path[:5]:  # 只显示前5个
#     print(f"  {path}")
