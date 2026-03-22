"""数据集加载器"""
from datasets import load_dataset
from typing import List, Dict


class GSM8KLoader:
    """GSM8K数据集加载器"""

    def __init__(self, split_ratio=(0.7, 0.15, 0.15)):
        self.dataset = load_dataset("gsm8k", "main")
        self.train_ratio, self.val_ratio, self.test_ratio = split_ratio

    def load_split(self, split="train"):
        """加载指定split的数据"""
        if split == "train":
            data = self.dataset["train"]
            n = int(len(data) * self.train_ratio)
            return [{"question": d["question"], "answer": d["answer"]}
                    for d in data[:n]]
        elif split == "val":
            data = self.dataset["train"]
            n_train = int(len(data) * self.train_ratio)
            n_val = int(len(data) * self.val_ratio)
            return [{"question": d["question"], "answer": d["answer"]}
                    for d in data[n_train:n_train+n_val]]
        elif split == "test":
            return [{"question": d["question"], "answer": d["answer"]}
                    for d in self.dataset["test"]]
        else:
            raise ValueError(f"Unknown split: {split}")
