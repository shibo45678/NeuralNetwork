project/

├── data/                   # 数据管理（新增：区分原始、中间、输出数据）
│   ├── raw/                # 原始数据（从业务系统导入，只读）
│   ├── intermediate/       # 中间数据（数据清洗后、特征工程前的中间态）
│   └── processed/          # 处理后数据（特征工程完成，供模型训练/预测）
├── src/                    # 源码目录（核心逻辑，按功能分层）
│   ├── data_cleaning/      # 改变样本数量的预处理（原data_cleaning模块，增强：配置驱动）
│   │   ├── outlier_removal.py  # 异常值处理（读取config/data_cleaning.yaml配置）
│   │   ├── duplicate_removal.py # 重复值处理（读取配置）
│   │   └── sampling.py     # 采样（读取配置，支持过采样/欠采样策略）
│   ├── feature_engineering/ # 不改变样本数量的转换（增强：配置+测试用例）
│   │   ├── scalers.py      # 标准化（如StandardScaler，支持配置选择方法）
│   │   ├── encoders.py     # 编码（如OneHot/LabelEncoder，支持配置编码字段）
│   │   ├── feature_generators.py # 特征生成（如时间sin/cos、滞后特征，配置驱动）
│   │   └── tests/          # 单元测试（确保特征转换逻辑正确）
│   ├── models/             # 模型（增强：版本管理+序列化）
│   │   ├── sklearn_models.py # sklearn模型（如LR、RandomForest，支持配置加载）
│   │   ├── neural_networks.py # 神经网络（如LSTM/CNN，封装成sklearn兼容接口）
│   │   ├── custom_models.py # 自定义模型（如融合模型）
│   │   └── model_store.py  # 模型序列化/加载（支持版本回滚）
│   ├── pipelines/          # 组合流程（增强：环境区分+监控埋点）
│   │   ├── training_pipeline.py # 训练Pipeline（数据清洗→特征工程→模型训练→保存）
│   │   ├── inference_pipeline.py # 推理Pipeline（特征工程→模型加载→预测）
│   │   ├── production_pipeline.py # 生产Pipeline（含监控、日志）
│   │   └── pipeline_utils.py # Pipeline工具类（如配置解析、组件注册）
│   └── utils/              # 工具库（通用功能）
│       ├── io_utils.py     # 数据读写（支持多种格式：CSV、Parquet、数据库）
│       ├── logging_utils.py # 日志（集成MLflow等监控工具）
│       └── config_utils.py # 配置解析（加载config目录下的yaml）
├── scripts/                # 执行脚本（新增：自动化任务入口）
│   ├── run_data_cleaning.py # 执行数据清洗（生成intermediate数据）
│   ├── run_feature_engineering.py # 执行特征工程（生成processed数据）
│   ├── train_model.py      # 训练模型（调用training_pipeline）
│   └── predict.py          # 推理（调用inference_pipeline）
├── tests/                  # 集成测试（确保端到端流程正确）
│   ├── test_data_pipeline.py # 数据流程测试（清洗→特征工程）
│   └── test_model_pipeline.py # 模型流程测试（训练→推理）
└── requirements.txt        # 依赖管理
