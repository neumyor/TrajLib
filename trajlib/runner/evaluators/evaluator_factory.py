from evaluators.base_evaluator import BaseEvaluator


def create_evaluator(model, dataset, config, device=None):
    """
    创建合适的Evaluator实例。

    :param model: 要评估的模型
    :param dataset: 数据集（可能是验证集或测试集）
    :param config: 配置文件，包含评估参数等
    :param device: 评估所使用的设备（CPU/GPU）
    :return: 返回合适的Evaluator实例
    """
    # 基于任务类型或者配置选择不同的评估器
    evaluator_type = config.get('evaluator_type', 'task')  # 默认为任务评估器

    if evaluator_type == 'task':
        pass
    else:
        print("Invalid evaluator type, using BaseEvaluator as default...")
        return BaseEvaluator(model, dataset, config, device)
