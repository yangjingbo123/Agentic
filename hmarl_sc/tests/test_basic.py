"""测试脚本 - 验证框架基本功能"""
import sys
sys.path.append('.')

from envs.reasoning_env import ReasoningEnv
from envs.blackboard import Blackboard, Message, MessageType
from models.high_policy import HighLevelPolicy
from models.low_policy import LowLevelPolicy
from models.value_nets import HighLevelValue, LowLevelValue
import torch
import yaml


def test_blackboard():
    """测试信息板"""
    print("测试Blackboard...")
    bb = Blackboard()
    msg = Message(sender=0, msg_type=MessageType.TRACE, content=("reasoning", "42"))
    bb.add_message(msg)
    assert len(bb.messages) == 1
    assert len(bb.traces) == 1
    print("✓ Blackboard测试通过")


def test_policies():
    """测试策略网络"""
    print("\n测试策略网络...")

    # 高层策略
    high_policy = HighLevelPolicy(obs_dim=271, hidden_dim=128)
    obs = torch.randn(2, 271)
    goal_logits, focus_logits, budget_logits = high_policy(obs)
    assert goal_logits.shape == (2, 5)
    assert budget_logits.shape == (2, 3)
    print("✓ 高层策略测试通过")

    # 低层策略
    low_policy = LowLevelPolicy(obs_dim=512, hidden_dim=128)
    obs = torch.randn(2, 512)
    work_logits, comm_logits, _ = low_policy(obs, "proposer")
    assert work_logits.shape == (2, 5)
    assert comm_logits.shape == (2, 5)
    print("✓ 低层策略测试通过")


def test_value_nets():
    """测试价值网络"""
    print("\n测试价值网络...")

    high_value = HighLevelValue(obs_dim=271, hidden_dim=128)
    obs = torch.randn(2, 271)
    value = high_value(obs)
    assert value.shape == (2, 1)
    print("✓ 价值网络测试通过")


def test_environment():
    """测试环境"""
    print("\n测试环境...")

    with open('configs/default.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    env = ReasoningEnv(config)
    obs = env.reset("What is 2+2?")
    assert "controller" in obs
    print("✓ 环境初始化测试通过")


if __name__ == "__main__":
    print("=" * 50)
    print("HMARL-SC 框架测试")
    print("=" * 50)

    test_blackboard()
    test_policies()
    test_value_nets()
    test_environment()

    print("\n" + "=" * 50)
    print("所有测试通过! ✓")
    print("=" * 50)
