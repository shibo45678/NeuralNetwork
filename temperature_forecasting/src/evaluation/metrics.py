import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from ..data.windows import WindowGenerator
import matplotlib.pyplot as plt
from typing import Dict
import tensorflow as tf


class ModelEvaluation:
    def __init__(self, output_configs: Dict, model_name: str = "cnn"):
        self.output_configs = output_configs
        self.model_name = model_name
        self.task_order = list(output_configs.keys())

    def comprehensive_model_evaluation(self,
                                       model: tf.keras.Model,
                                       window: 'WindowGenerator',
                                       valsets,
                                       testsets,
                                       ) -> Dict:
        # 1. 基础评估
        print("=" * 60)
        print(f"开始评估 {self.model_name}")
        print("=" * 60)

        task_metrics = self._evaluate_multi_task_model(model, window, valsets, testsets, )

        val_analysis = self._detailed_multi_task_evaluation(model, window, valsets, "验证集")
        test_analysis = self._detailed_multi_task_evaluation(model, window, testsets, "测试集")

        self._print_summary_report(task_metrics)

        return {
            'task_metrics': task_metrics,  # 基础指标
            'val_analysis': val_analysis,  # 验证集详细分析
            'test_analysis': test_analysis,  # 测试集详细分析
            'model_name': self.model_name
        }

    def _print_summary_report(self, task_metrics: Dict):
        print("\n" + "=" * 60)
        print(f"{self.model_name} - 评估汇总报告")
        print("=" * 60)

        for task_name, metrics in task_metrics.items():
            task_type = metrics['type']
            if task_type == 'regression':
                print(f"{task_name}: Val_MAE = {metrics['val_metric']:.4f},Test_MAE = {metrics['test_metric']:.4f}")
            else:
                print(
                    f"{task_name}: Val_Accuracy={metrics['val_metric']:.4f}, Test_Accuracy={metrics['test_metric']:.4f}")

    def _evaluate_multi_task_model(self, model, window, valsets, testsets) -> Dict:
        """混合分类和回归"""
        """
        评估多任务模型（混合回归和分类）
        Args:
            model: 已训练的Keras模型
            window: 窗口生成器对象
        """

        # 绘制预测结果
        if hasattr(window, 'window_plot'):
            window.window_plot(model)
            plt.show()

        print(f"模型指标：{model.metrics_names}")

        # 评估模型
        val_performance = model.evaluate(valsets, verbose=0)  # 所有损失和指标的'数值'列表
        test_performance = model.evaluate(testsets, verbose=0)

        # 解析评估结果
        val_metrics = dict(zip(model.metrics_names, val_performance))  # 键，上面的数值
        test_metrics = dict(zip(model.metrics_names, test_performance))
        # {
        #     'loss': 0.25,                    # 总损失
        #     'output_temperature_loss': 0.15, # 温度任务损失
        #     'output_temperature_mae': 0.12,  # 温度任务MAE
        #     'output_weather_type_loss': 0.10,# 天气类型任务损失
        #     'output_weather_type_accuracy': 0.85 # 天气类型准确率
        # }

        print(f"\n=== {self.model_name} 模型评估结果 ===")
        print("验证集:", val_metrics)
        print("测试集:", test_metrics)

        # 为每个任务单独计算指标
        task_metrics = {}
        for task_name, config in self.output_configs.items():
            task_type = config['type']
            output_layer_name = f'output_{task_name}'

            print(f"\n--- 任务: {task_name} ({task_type}) ---")

            val_loss = val_metrics.get(f'{output_layer_name}_loss', 0)
            test_loss = test_metrics.get(f'{output_layer_name}_loss', 0)

            if task_type == 'regression':
                val_metric = val_metrics.get(f'{output_layer_name}_mae', 0)
                test_metric = test_metrics.get(f'{output_layer_name}_mae', 0)
                metric_name = 'MAE'
            else:  # binary_classification + 多分类
                val_metric = val_metrics.get(f'{output_layer_name}_accuracy', 0)
                test_metric = test_metrics.get(f'{output_layer_name}_accuracy', 0)
                metric_name = 'Accuracy'

            print(f"{task_name} - 验证集{metric_name}: {val_metric:.4f}, 测试集{metric_name}: {test_metric:.4f}")

            # 存储任务指标
            task_metrics[task_name] = {
                'val_loss': val_loss,
                'test_loss': test_loss,
                'val_metric': val_metric,
                'test_metric': test_metric,
                'type': task_type
            }

        return task_metrics

    def _detailed_multi_task_evaluation(self,
                                        model: tf.keras.Model,
                                        window: 'WindowGenerator',
                                        dataset: tf.data.Dataset,
                                        dataset_type: str = "验证集") -> Dict:

        # 获取一批数据进行详细分析
        inputs, true_labels = next(iter(dataset))
        predictions = model.predict(inputs, verbose=0)

        print(f"\n==={self.model_name} - {dataset_type}详细分析 ===")
        task_results = {}

        # 多输出模型：predictions是元组
        if isinstance(predictions, (tuple, list)):
            print(f"\n=== {self.model_name} - {dataset_type}详细分析 ===")
            print(f"inputs形状: {inputs.shape}")
            print(f"true_labels形状: {true_labels.shape}")
            print(f"predictions类型: {type(predictions)}")

            for i, (task_name, config) in enumerate(self.output_configs.items()):
                if i < len(predictions):
                    task_type = config['type']
                    pred = predictions[i]  # 第i个输出层的预测
                    true = true_labels[:, :, i]  # 从合并标签中提取对应任务
                    task_results[task_name] = self._analyze_single_task(pred, true, config, task_name)

        return task_results

    def _analyze_single_task(self, predictions: np.ndarray,
                             true_values: np.ndarray,
                             config: Dict,
                             task_name: str) -> Dict:

        task_type = config['type']
        print(f"\n--- 任务: {task_name} ({task_type}) ---")

        if task_type == 'regression':
            return self._analyze_regression_task(predictions, true_values, task_name)

        elif task_type == 'binary_classification':
            # 二分类分析
            return self._analyze_binary_classification_task(predictions, true_values, task_name)

        elif task_type == 'classification':
            # 多分类分析
            return self._analyze_multiclass_task(predictions, true_values, task_name, config.get('num_classes', 3))
        else:
            print(f"未知任务类型: {task_type}")
            return {}

    def _analyze_regression_task(self, predictions: np.ndarray,
                                 true_values: np.ndarray,
                                 task_name) -> Dict:

        # 确保数据形状正确
        predictions = predictions.squeeze()
        true_values = true_values.squeeze()

        mae = np.mean(np.abs(predictions - true_values))
        mse = np.mean((predictions - true_values) ** 2)
        rmse = np.sqrt(mse)

        print(f"-----{task_name}-----")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")

        # 预测值统计
        print(f"预测值范围: [{predictions.min():.3f}, {predictions.max():.3f}]")
        print(f"真实值范围: [{true_values.min():.3f}, {true_values.max():.3f}]")

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'predictions': predictions,
            'true_values': true_values
        }

    def _analyze_binary_classification_task(self, predictions: np.ndarray,
                                            true_values: np.ndarray,
                                            task_name: str) -> Dict:
        """分析二分类任务"""
        pred_probs = predictions.squeeze()  # 概率值
        true_binary = true_values.squeeze().astype(int)

        pred_binary = (pred_probs > 0.5).astype(int)
        accuracy = np.mean(pred_binary == true_binary)  # 准确率 = 正确预测的样本数 / 总样本数

        print(f"Accuracy: {accuracy:.4f}")
        print("分类报告:")
        print(classification_report(true_binary, pred_binary, zero_division=0))

        # 混淆矩阵
        cm = confusion_matrix(true_binary, pred_binary)
        print("混淆矩阵:")
        print(cm)

        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'predictions': pred_binary,
            'probabilities': pred_probs,
            'true_values': true_binary
        }

    def _analyze_multiclass_task(self, predictions: np.ndarray,
                                 true_values: np.ndarray,
                                 task_name: str,
                                 num_classes: int) -> Dict:

        """分析多分类任务"""

        pred_probs = predictions
        pred_classes = np.argmax(predictions, axis=-1)
        # np.argmax() 返回数组中最大值的索引 ,每个样本中最大概率的索引
        # 样本1: max(0.1, 0.8, 0.1) = 0.8 → 索引1
        # 样本2: max(0.7, 0.2, 0.1) = 0.7 → 索引0

        true_classes = true_values.squeeze().astype(int)
        accuracy = np.mean(pred_classes == true_classes)

        print(f"Accuracy: {accuracy:.4f}")
        print("分类报告:")
        print(classification_report(true_classes, pred_classes, zero_division=0))

        # 混淆矩阵
        cm = confusion_matrix(true_classes, pred_classes)
        print("混淆矩阵:")
        print(cm)

        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'predictions': pred_classes,
            'probabilities': pred_probs,
            'true_values': true_classes
        }
