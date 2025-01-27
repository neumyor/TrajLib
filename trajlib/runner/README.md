1. user指定任务类型（Task）、数据集（Dataset）、模型（Model）
2. 根据Task找到对应的数据集读取方式（DataLoader）
3. 根据模型找到对应的运行器（Runner），然后把参数传递给它
   -  Runner中定义了Model的训练流程
      -  如果Model仅需单阶段训练，则包含一个Trainer
      -  如果Model包含多阶段训练，则包含多个Trainer
      -  每个Trainer独立构成一个完整的训练过程：
         -  独立的dataset和dataloader设置
         -  独立的train/validate 学习参数设置
         -  Trainer的Test方法外包到Evaluator实现
         -  如果Model有多个下游任务，则包含多个Evaluator
4. Runner的输入是参数，输出是模型的训练ckpt和日志