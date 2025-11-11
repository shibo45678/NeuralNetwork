import os
from typing import List, Dict, Union

import numpy as np
import pandas as pd
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
    configs: List[ProcessorConfig] = Field(..., description="处理器配置列表")  # 保证了列表存在

    @model_validator(mode='after')
    def validate_cross_config_overlap(self):
        """验证不同配置项间的重复实例"""
        all_instances = []
        duplicate_instances = []

        for config in self.configs:
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
        return cls(configs=config_list)

    def __iter__(self):
        return iter(self.configs)

    def __getitem__(self, index):
        return self.configs[index]

    def __len__(self):
        return len(self.configs)


# 协调器
class CompletePreprocessor:

    def __init__(self, processor_configs: Union[List[Dict], ProcessorConfigs]):
        self.processor_configs = self._initialize_processor_config(processor_configs)
        self.pipelines = {}
        self.fitted_cleaners = {}  # 拟合的cleaner也记录下，防止预测是processor_configs改变

    def _initialize_processor_config(self, processor_configs) -> ProcessorConfigs:
        if processor_configs is None:
            return ProcessorConfigs(configs=[])
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

        for idx, part in enumerate(self.processor_configs):
            print(f"实际类型: {type(part)}")
            print(f"是否是字典: {isinstance(part, dict)}")
            print(f"是否是ProcessorConfig: {isinstance(part, ProcessorConfig)}")

            class_obj_list = part.obj_list
            change = part.len_change

            if not class_obj_list:
                continue  # 跳过空配置

            if change:
                # 记录一下用过的cleaner里面的类的实例
                fitted_cleaners = []

                for cleaner in class_obj_list:
                    old = len(features_temp)

                    features_temp, labels_temp = cleaner.learn_process(features_temp, labels_temp)
                    fitted_cleaners.append(cleaner)

                    new = len(features_temp)
                    print(f"第{idx + 1}步完成: {old} -> {new} 样本")

                self.fitted_cleaners[f'cleaner_{idx + 1}'] = fitted_cleaners  # 阶段包含多个cleaners

            else:
                # 创建pipeline 并缓存
                cachedir = f'./pipeline_cache/pipeline_{idx + 1}'
                if not os.path.exists(cachedir):
                    os.makedirs(cachedir)

                memory = Memory(location=cachedir, verbose=0)

                steps = [(f'engineer_{i + 1}', engineer) for i, engineer in enumerate(class_obj_list)]

                pipeline = Pipeline(steps, memory=memory)

                old = features_temp.shape
                features_temp = pipeline.fit_transform(features_temp, labels_temp)
                new = features_temp.shape
                print(f"第{idx + 1}步完成: 生成 pipeline，sklearn pipeline不改变数据形状。数据形状:{old} -> {new}")

                self.pipelines[f'pipeline_{idx + 1}'] = pipeline

        return features_temp, labels_temp

    def transform(self, features, labels=None):
        if not self.fitted_cleaners and not self.pipelines:
            logger.info("请先调用 train() 方法，获取必要的训练数据")
            return raw_data, labels

        features_temp, labels_temp = features, labels

        for idx, part in enumerate(self.processor_configs):

            change = part.len_change
            class_obj_list = part.obj_list
            if not class_obj_list:
                continue

            if change:
                # 使用已训练的cleaners进行转换
                for cleaner in self.fitted_cleaners.get(f'cleaner_{idx}', []):
                    features_temp, labels_temp = cleaner.learn_process(features_temp, labels_temp)
            else:
                pipeline = self.pipelines.get(f'pipeline_{idx}')
                if pipeline:
                    features_temp = pipeline.transform(features_temp)

        return features_temp, labels_temp

    def get_all_attributes(self):
        all_attributes = {}  # {pipeline_1: steps_info}  steps_info={step_name:[attributes]}
        for name, pipeline in self.pipelines.items():
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
            step = self.pipelines.get(f'pipeline_{idx}').named_steps[step_name]
            return getattr(step, attribute_name)
        except(KeyError, AttributeError) as e:
            print(f"获取属性失败: {e}")
            return None



if __name__ == '__main__':
    d = {'feature1': np.random.normal(0, 1, 9),
         'feature2': ['a', 'b', 'c', 'c', 'a', 'b', 'c', 'a', 'b']}
    raw_data = pd.DataFrame(d)
    labels = pd.Series([0, 1, 1, 1, 1, 3, 6, 7, 8])

    configs2 = [{'obj_list': [RemoveDuplicates()], 'len_change': True},
                {'obj_list': [UnifiedFeatureScaler(algorithm='cnn'),
                              CategoricalEncoding(handle_unknown='ignore', unknown_token='__UNKNOWN__')],
                 'len_change': False}]
    obj = CompletePreprocessor(configs2)
    obj.train(raw_data, labels)
    all_attributes = obj.get_all_attributes()
    print(all_attributes)

    b = obj.get_specific_attribute(2, 'engineer_1', 'numeric_columns_')
    print(f'应该结果是：[feature1] ，结果{b}')

    a = obj.get_specific_attribute(2, 'engineer_2', 'categorical_columns_')
    print(f'应该结果是：[feature2] ，结果{a}')

    # features_temp, labels_temp = obj.transform(raw_data, labels)
