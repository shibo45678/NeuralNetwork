import copy
import os
import warnings
from typing import List, Dict, Union

import numpy as np
import pandas as pd
import pytest
from joblib import Memory
from pydantic import BaseModel, field_validator, model_validator, Field
from sklearn.pipeline import Pipeline
import logging
from data.data_preparation.remove_duplicates import RemoveDuplicates
from data.feature_engineering import UnifiedFeatureScaler, CategoricalEncoding

logger = logging.getLogger(__name__)

'''
    1. 严格按照数据的【处理顺序】使用'obj_list'，并标记'len_change'(这里将改变数据长度的步骤，手动处理）
    2. 手动处理的类:是无法放进pipeline的类，不会继承BaseEstimator和TransfromerMixin。并且使用learn_process处理。
    [
        {'obj_list':[DescribeData()], 'len_change':False },
        {'obj_list':[RemoveDuplicates(), DeleteUselessCols()], 'len_change':True},
        {'obj_list':[NumericMissingValueHandler(),CategoricalMissingValueHandler()],'len_change':False},
        {'obj_list':[Resampler()],'len_change':True},
    ]
    
    3.[ProcessorConfigs('obj_list':,'len_change)]
       差异大的检查：不阻止相同类+相同参数的不同实例，阻止同一实例： sampler = Resampler() 'obj_list': [sampler,sampler]
       差异小的检查：防止写成类Resampler，非实例Resampler() 丢括号
'''


class ProcessorConfig(BaseModel):  # 对应数据结构中的一个字典
    obj_list: List[object] = Field(..., description="处理器实例列表")  # 这个字段是必需的，且没有默认值。
    len_change: bool = Field(default=True, description="是否改变数据长度")

    @field_validator('obj_list')
    @classmethod
    def validate_obj_list(cls, v):
        if not v or not isinstance(v, List):
            error_msg = f"{v}字段必须是列表格式且不为空"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 避免是 类而非实例()
        if any(isinstance(obj, type) for obj in v):  # 存在任何一个是类本身就需要报错
            raise ValueError(f"obj_list字段元素必须为类的实例，不是类本身")
        return v

    @field_validator('len_change')
    @classmethod
    def validate_len_change(cls, v):
        if not isinstance(v, bool):
            raise ValueError(f"{v}字段必须是布尔值True/False")
        return v

    @model_validator(mode='after')
    def _validate_instance_methods_and_overlap(self):
        """验证方法存在性和同一行内重复实例"""
        current_instances = self.obj_list
        # 只检查同一行内的重复实例
        seen_ids = set()
        duplicates = []
        for inst in current_instances:
            inst_id = id(inst)
            if inst_id in seen_ids:
                duplicates.append(inst)
            else:
                seen_ids.add(inst_id)
        if duplicates:
            raise ValueError(f"配置内存在重复实例: {duplicates}")

        # 验证方法是否存在
        if self.len_change:
            for instance in current_instances:
                if not hasattr(instance, 'learn_process'):
                    raise ValueError(f" 实例 {instance} 缺少 learn_process 方法")
        else:
            for instance in current_instances:
                # 检查是否具有必要的方法
                if not hasattr(instance, 'fit'):
                    raise ValueError(f"实例 {instance} 缺少 fit 方法")
                if not hasattr(instance, 'transform'):
                    raise ValueError(f"实例 {instance} 缺少 transform 方法")
                if not callable(getattr(instance, 'fit', None)):
                    raise ValueError(f"实例 {instance} 的 fit 不是可调用方法")  # 拦截字符串None等非可调用方法
        return self


class ProcessorConfigs(BaseModel):
    """完整的处理器配置列表"""
    configs_: List[ProcessorConfig] = Field(..., description="处理器配置列表")  # 保证了列表存在

    @model_validator(mode='after')
    def validate_cross_config_overlap(self):
        """验证不同配置项间的重复实例"""
        all_instances = []
        duplicate_instances = []

        for config in self.configs_:
            for instance in config.obj_list:
                instance_id = id(instance)
                if instance_id in [id(existing) for existing in all_instances]:
                    duplicate_instances.append(instance)
                all_instances.append(instance)

        if duplicate_instances:
            raise ValueError(f"发现跨配置重复使用的实例: {set(duplicate_instances)}")

        return self

    @classmethod
    def from_list(cls, config_list: List):
        """从列表创建配置"""
        return cls(configs_=config_list)

    def __iter__(self):
        return iter(self.configs)

    def __getitem__(self, index):
        return self.configs[index]

    def __len__(self):
        return len(self.configs)


# 协调器
class CompletePreprocessor:

    def __init__(self, processor_configs: Union[List[Dict], ProcessorConfigs]):
        self.processor_configs = processor_configs
        self.pipelines_ = {}
        self.validate_processor_configs_ = self._get_processor_configs()
        self.processor_classes_ = {}  # 保存处理器类信息

    def _get_processor_configs(self):
        return self._initialize_processor_config(self.processor_configs)

    def _initialize_processor_config(self, processor_configs) -> ProcessorConfigs:
        if processor_configs is None:
            return ProcessorConfigs(configs_=[])
        elif isinstance(processor_configs, ProcessorConfigs):
            return processor_configs
        elif isinstance(processor_configs, list):
            return ProcessorConfigs.from_list(processor_configs)
        else:
            error_msg = f"processor_configs 必须是列表结构 、ProcessorConfigs实例或None"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def train(self, features, labels=None):
        features_temp, labels_temp = features, labels

        for idx, part in enumerate(self.validate_processor_configs_.configs_):
            logger.info(f"实际类型: {type(part)}")
            logger.info(f"是否是字典: {isinstance(part, dict)}")
            logger.info(f"是否是ProcessorConfig: {isinstance(part, ProcessorConfig)}")

            class_obj_list = part.obj_list
            change = part.len_change

            if not class_obj_list:
                continue  # 跳过空配置

            if change:
                # 保存cleaner初始化参数等信息
                processor_info_list = []

                for i, cleaner in enumerate(class_obj_list):
                    old = len(features_temp)
                    features_temp, labels_temp = cleaner.learn_process(features_temp, labels_temp)
                    new = len(features_temp)

                    # 保存处理类信息和初始化参数（用于后续创建新实例）
                    processor_info = {
                        'class': type(cleaner),
                        'init_params': getattr(cleaner, '_init_params', {}),  # 如果有初始化参数
                        # 'trained_state': getattr(cleaner, 'get_state', lambda: {})(),  # 获取训练状态（暂无用，兜底lambda）
                    }

                    processor_info_list.append(processor_info)
                    logger.info(f"stage{idx}，第{i}步完成: {old} -> {new} 样本")

                self.processor_classes_[f'cleaner_{idx}'] = processor_info_list


            else:
                # 创建pipeline 并缓存
                # cachedir = f'./pipeline_cache/pipeline_{idx + 1}'
                # if not os.path.exists(cachedir):
                #     os.makedirs(cachedir)
                #
                # memory = Memory(location=cachedir, verbose=0)

                steps = [(f'engineer_{i}', engineer) for i, engineer in enumerate(class_obj_list)]

                pipeline = Pipeline(steps,)# memory=memory

                old = features_temp.shape
                features_temp = pipeline.fit_transform(features_temp, labels_temp)
                new = features_temp.shape

                # 添加详细的调试信息
                logger.info(f"Pipeline {idx} 步骤: {[name for name, _ in steps]}")

                # 检查每个转换器的训练状态
                for step_name, transformer in pipeline.steps:
                    if hasattr(transformer, 'n_features_in_'):
                        logger.info(f"  {step_name}: 训练特征数 = {transformer.n_features_in_}")
                    if hasattr(transformer, 'classes_'):
                        logger.info(f"  {step_name}: 类别数 = {len(transformer.classes_)}")

                self.pipelines_[f'pipeline_{idx}'] = pipeline


                logger.info(
                    f"stage{idx}完成: 生成 pipeline，sklearn pipeline不改变数据形状。数据形状:{old} -> {new}")

                self.pipelines_[f'pipeline_{idx}'] = pipeline

        return features_temp, labels_temp

    def transform_predict(self, features, labels=None):

        if not self.processor_classes_ and not self.pipelines_:
            logger.info("请先调用 train() 方法，获取必要的训练数据")
            return features, labels

        features_temp, labels_temp = features, labels

        for idx, part in enumerate(self.validate_processor_configs_.configs_):

            change = part.len_change
            class_obj_list = part.obj_list
            if not class_obj_list:
                continue

            if change:
                # 使用新的cleaners实例进行转换
                cleaners_info_list = self.processor_classes_.get(f'cleaner_{idx}',[])
                for cleaner in cleaners_info_list:
                    # 获取该类新实例
                    cleaner_class = cleaner.get('class')
                    cleaner_init_params = cleaner.get('init_params',{})

                    new_cleaner = cleaner_class(**cleaner_init_params)
                    features_temp, labels_temp = new_cleaner.learn_process(features_temp, labels_temp)

            else:
                pipeline = self.pipelines_.get(f'pipeline_{idx}')
                if pipeline:
                    features_temp = pipeline.transform(features_temp)

        return features_temp, labels_temp

    def get_all_attributes(self):
        all_attributes = {}  # {pipeline_1: steps_info}  steps_info={step_name:[attributes]}
        for name, pipeline in self.pipelines_.items():
            all_attributes[name] = self._get_all_steps_info(pipeline)
        return all_attributes

    def _get_all_steps_info(self, pipeline):
        steps_info = {}
        for step_name in pipeline.named_steps.keys():
            steps_info[step_name] = self._get_step_attributes(pipeline, step_name)
        return steps_info

    def _get_step_attributes(self, pipeline, step_name):

        step = pipeline.named_steps[step_name]
        attributes = {}

        # 获取所有以下划线结尾的属性（sklearn 的惯例）
        for attr_name in dir(step):
            if attr_name.endswith('_') and not attr_name.startswith('_'):
                attr_value = getattr(step, attr_name)
                attributes[attr_name] = attr_value
        return attributes

    def get_specific_attribute(self, idx, step_name, attribute_name):
        """获取指定步骤的特定属性"""
        try:
            step = self.pipelines_.get(f'pipeline_{idx}').named_steps[step_name]
            return getattr(step, attribute_name)
        except(KeyError, AttributeError) as e:
            warnings.warn(f"获取属性失败: {e}")
            return None


def test_completepreprocessor():
    np.random.seed(42)
    d = {'feature1': np.random.randint(0, 100, 9),
         'feature2': ['a', 'b', 'c', 'c', 'a', 'b', 'c', 'a', 'b']}
    raw_data = pd.DataFrame(d)
    labels = pd.Series([0, 1, 1, 1, 1, 3, 6, 7, 8])

    new_data = pd.DataFrame({'feature1': [2, 1, 1, 2, 3],
                             'feature2':['a','b','b','c','d']})
    new_labels = pd.Series([0, 1, 1, 1, 1])

    config = {
        'transformers': [{'standard': {'columns': ['feature1']}}],
        'skip_scale': ['feature2']}

    configs2 = [{'obj_list': [RemoveDuplicates()], 'len_change': True},
                {'obj_list': [UnifiedFeatureScaler(method_config=config, algorithm='cnn'),
                              ], # CategoricalEncoding(handle_unknown='ignore', unknown_token='__UNKNOWN__')
                 'len_change': False}]

    obj = CompletePreprocessor(configs2)
    result=obj.train(raw_data, labels)
    new_result = obj.transform_predict(new_data, new_labels)

    logger.debug("转换的训练数据应该：")
    with pytest.raises(ValueError):
        assert len(result) == 4

    logger.debug("转换的新数据不应该为空：")
    assert new_result is not None


    logger.debug("转换的新数据应该要能去重：")
    with pytest.raises(ValueError):
        assert len(new_result) == 4

    logger.debug("可以提取特定步骤的属性")
    all_attributes = obj.get_all_attributes()
    assert 'numeric_columns_' in all_attributes

    b = obj.get_specific_attribute('1', 'engineer_0', 'numeric_columns_')
    logger.debug(f'new_data应该结果是：[feature1] ，结果{b}')
    assert b == ['feature1']


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
