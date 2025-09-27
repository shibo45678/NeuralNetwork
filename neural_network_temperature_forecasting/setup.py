# setup.py
from setuptools import setup, find_packages

setup(
    name="neural_network_temperature_forecasting",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.10",
        "numpy>=1.21",
        "pandas>=1.3",
        "matplotlib>=3.5",
    ],
)