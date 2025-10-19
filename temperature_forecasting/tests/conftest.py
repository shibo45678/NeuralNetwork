# /Users/shibo/Python/NeuralNetwork/temperature_forecasting/tests/conftest.py
import sys
import os

# 在模块级别设置路径（确保在导入测试文件前执行）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')

if src_path not in sys.path:
    sys.path.insert(0, src_path)
    print(f"✅ conftest.py 已设置路径: {src_path}")


def pytest_configure(config):
    """pytest 配置钩子 - 可选"""
    print("✅ pytest 配置完成")
