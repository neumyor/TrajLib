# 联邦轨迹聚类
## 1. 安装依赖
```
conda create -f py310.yaml -n fedtraj 
```
## 2. 运行流程
1. preprocess_porto.ipynb 读取proto数据集并转换为pkl文件（该文件会被存放在fedtraj/data下）
2. federated_embedding.ipynb 训练模型（暂定，轨迹段切分的实现放在了fedtraj.model.trainer.utils.py中，在模型训练前会切分）
3. federated_clustering.ipynb 聚类并计算结果（包含凝聚系数SC）

