"""Stage 0: 行为克隆初始化"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import yaml
from tqdm import tqdm


class BCDataset(Dataset):
    """BC训练数据集"""

    def __init__(self, trajectories: List):
        self.high_data = []
        self.low_data = []

        for traj in trajectories:
            for level, obs, action in traj:
                if level == "high":
                    self.high_data.append((obs, action))
                else:
                    self.low_data.append((obs, action))

    def __len__(self):
        return len(self.high_data) + len(self.low_data)

    def get_high_data(self):
        return self.high_data

    def get_low_data(self):
        return self.low_data


def train_bc(high_policy, low_policy, trajectories, config):
    """BC训练主函数"""
    dataset = BCDataset(trajectories)
    epochs = config.get("bc_epochs", 30)
    batch_size = config.get("bc_batch_size", 32)
    lr = config.get("bc_lr", 1e-3)

    optimizer_high = torch.optim.Adam(high_policy.parameters(), lr=lr)
    optimizer_low = torch.optim.Adam(low_policy.parameters(), lr=lr)

    print(f"BC训练: {len(dataset.get_high_data())} 高层样本, {len(dataset.get_low_data())} 低层样本")

    for epoch in range(epochs):
        # 训练高层
        high_loss = train_high_epoch(high_policy, optimizer_high,
                                     dataset.get_high_data(), batch_size)

        # 训练低层
        low_loss = train_low_epoch(low_policy, optimizer_low,
                                   dataset.get_low_data(), batch_size)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - High Loss: {high_loss:.4f}, Low Loss: {low_loss:.4f}")

    return high_policy, low_policy


def train_high_epoch(policy, optimizer, data, batch_size):
    """训练高层一个epoch"""
    policy.train()
    total_loss = 0

    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        obs_batch = torch.stack([torch.tensor(obs) for obs, _ in batch])

        optimizer.zero_grad()
        goal_logits, _, budget_logits = policy(obs_batch)

        # 简化的loss计算
        loss = goal_logits.mean() + budget_logits.mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(data) // batch_size, 1)


def train_low_epoch(policy, optimizer, data, batch_size):
    """训练低层一个epoch"""
    policy.train()
    total_loss = 0

    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        obs_batch = torch.stack([torch.tensor(obs) for obs, _ in batch])

        optimizer.zero_grad()

        # 对三个角色分别训练
        loss = 0
        for role in ["proposer", "critic", "verifier"]:
            work_logits, comm_logits, _ = policy(obs_batch, role)
            loss += work_logits.mean() + comm_logits.mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(data) // batch_size, 1)
