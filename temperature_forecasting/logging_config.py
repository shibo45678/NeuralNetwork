import logging
import logging.config

LOGGING_CONFIG = {
    'version': 1,  # 必须为1
    'disable_existing_loggers': False,  # 不禁用已存在的logger

    # 1. 定义输出格式（日志消息的外观）
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            # 显示：时间 - 模块名 - 级别 - 消息
        }
    },

    # 2. 定义输出目的地（日志输出到哪里）
    'handlers': {
        'default': {
            'level': 'INFO',  # 处理INFO及以上级别的消息
            'class': 'logging.StreamHandler',  # 输出到控制台
            'formatter': 'standard'  # 使用上面的格式
        }
    },

    # 3. 定义不同模块的日志行为
    'loggers': {
        '': {  # 根logger - 所有模块的默认配置
            'handlers': ['default'],
            'level': 'INFO',  # 默认只显示INFO及以上
            'propagate': True  # 向上传递日志消息
        },
        # 'data.transformers': {  # 专门为data.transformers模块配置
        #     'handlers': ['default'],
        #     'level': 'DEBUG',  # 显示DEBUG及以上（更详细）
        #     'propagate': False  # 不向上传递，避免重复
        # },
        'sklearn': {  # 第三方库sklearn的配置
            'handlers': ['default'],
            'level': 'WARNING',  # 只显示WARNING及以上（减少噪音）
            'propagate': False
        }
    }
}

if __name__== "__main__":
    # test_logging.py
    import logging.config
    import logging_config

    # 设置日志
    logging.config.dictConfig(logging_config.LOGGING_CONFIG)

    # 测试不同模块的日志
    logger1 = logging.getLogger("data.transformers")
    logger2 = logging.getLogger("other.module")
    logger3 = logging.getLogger("sklearn")

    logger1.debug("这是data.transformers的DEBUG信息")  # 如果配置了DEBUG会显示
    logger1.info("这是data.transformers的INFO信息")  # 会显示
    logger2.info("这是其他模块的INFO信息")  # 会显示
    logger2.debug("这是其他模块的DEBUG信息")  # 不会显示（默认INFO级别）
    logger3.debug("sklearn的DEBUG信息")  # 不会显示（配置为WARNING）