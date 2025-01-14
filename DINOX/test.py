from datasets import load_dataset
dataset = load_dataset("nyu-visionx/VSI-Bench")
print(dataset)  # 查看数据集结构
sample = dataset["test"][0]  # 查看一个样本的结构
print(sample)