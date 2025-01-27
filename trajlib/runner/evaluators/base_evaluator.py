import torch
from torch.utils.data import DataLoader

class BaseEvaluator:
    def __init__(self, model, dataset, config, device=None):
        """
        初始化评估器。

        :param model: 需要评估的模型
        :param dataset: 测试数据集
        :param config: 配置文件，包括评估超参数等
        :param device: 评估所使用的设备（CPU/GPU）
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.dataloader = DataLoader(self.dataset, batch_size=self.config['batch_size'], shuffle=False)
        self.criterion = torch.nn.CrossEntropyLoss()  # 这里以分类任务为例，损失函数可自定义

    def _evaluate_one_epoch(self):
        """
        执行单个epoch的评估。

        :return: 平均损失和准确率
        """
        self.model.eval()  # 设置模型为评估模式
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():  # 评估时不需要计算梯度
            for batch_idx, (inputs, labels) in enumerate(self.dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        epoch_loss = running_loss / len(self.dataloader)
        epoch_accuracy = correct / total
        return epoch_loss, epoch_accuracy

    def evaluate(self):
        """
        启动评估过程，进行一个完整的评估。
        """
        eval_loss, eval_accuracy = self._evaluate_one_epoch()
  
