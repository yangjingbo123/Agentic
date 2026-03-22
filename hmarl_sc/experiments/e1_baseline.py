"""E1实验: 基础框架 (仅Proposer + Controller)"""
import yaml


E1_CONFIG = {
    "experiment": "e1_baseline",
    "description": "基础框架 - 仅Proposer生成推理链",

    # 继承默认配置
    "base_config": "../configs/default.yaml",

    # E1特定配置
    "agents": {
        "proposer": {"enabled": True},
        "critic": {"enabled": False},
        "verifier": {"enabled": False},
    },

    "high_level": {
        "goals": ["EXPLORE-open", "STOP"],  # 只允许EXPLORE和STOP
        "aggregation": "majority",
    },

    "evaluation": {
        "baseline": ["SC-5", "SC-10", "SC-20"],
        "metrics": ["accuracy", "avg_cost", "k_equiv", "stop_rounds"],
    }
}


if __name__ == "__main__":
    with open("e1_baseline.yaml", "w", encoding="utf-8") as f:
        yaml.dump(E1_CONFIG, f, allow_unicode=True)
    print("E1配置已生成")
