from easydict import EasyDict


class ModelConfig(EasyDict):
    def __init__(self):
        super(ModelConfig, self).__init__()

        # Model architecture configuration
        self.input_size = 128  # 输入大小
        self.hidden_size = 256  # 隐藏层大小
        self.output_size = 10  # 输出类别数
        self.num_layers = 2  # LSTM 层数
        self.dropout = 0.5  # Dropout 概率

        # 网络模型的其他超参数
        self.activation_function = "ReLU"  # 激活函数类型
        self.batch_norm = True  # 是否使用BatchNorm

        # Model specific configurations (you can extend this based on your model)
        self.use_pretrained = False  # 是否使用预训练权重


class DatasetConfig(EasyDict):
    def __init__(self):
        super(DatasetConfig, self).__init__()

        # Dataset paths
        self.train_data_path = "./data/train"  # 训练集路径
        self.val_data_path = "./data/val"  # 验证集路径
        self.test_data_path = "./data/test"  # 测试集路径

        # Dataset specific configurations
        self.batch_size = 32  # 批大小
        self.shuffle = True  # 是否打乱数据
        self.num_workers = 4  # 数据加载时的工作线程数
        self.pin_memory = True  # 是否将数据加载到固定内存中以加速GPU训练

        # 数据预处理配置
        self.resize = (256, 256)  # 图像尺寸调整
        self.normalization_mean = [0.485, 0.456, 0.406]  # 图像均值
        self.normalization_std = [0.229, 0.224, 0.225]  # 图像标准差


class TrainingConfig(EasyDict):
    def __init__(self):
        super(TrainingConfig, self).__init__()

        # Training settings
        self.batch_size = 64  # 批大小
        self.learning_rate = 1e-3  # 学习率
        self.num_epochs = 20  # 训练轮数
        self.optimizer = "adam"  # 优化器类型
        self.loss_function = "cross_entropy"  # 损失函数类型
        self.lr_scheduler = "step_lr"  # 学习率调度器

        # Advanced training configurations
        self.weight_decay = 1e-5  # 权重衰减
        self.momentum = 0.9  # 动量

        # Logging and Checkpoints
        self.log_interval = 10  # 每多少步骤记录一次日志
        self.checkpoint_dir = "./checkpoints"  # 模型保存路径
        self.save_model = True  # 是否保存模型
        self.resume_from_checkpoint = False  # 是否从检查点恢复训练
