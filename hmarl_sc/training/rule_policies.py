"""规则策略 - 用于BC数据收集"""


class RulePolicy:
    """规则策略基类"""

    def get_high_level_action(self, state, round_num):
        """返回高层动作 (goal, focus, budget)"""
        raise NotImplementedError

    def get_low_level_actions(self, goal, focus, blackboard, step):
        """返回低层三agent的动作"""
        raise NotImplementedError


class SimpleRule(RulePolicy):
    """简单规则: EXPLORE×3 → STOP(majority)"""

    def get_high_level_action(self, state, round_num):
        if round_num < 3:
            return ("EXPLORE", "open", "standard")
        else:
            return ("STOP", "majority", None)

    def get_low_level_actions(self, goal, focus, blackboard, step):
        if step == 0:
            return {
                "proposer": ("generate", "submit-trace", None),
                "critic": ("work-idle", "comm-idle", None),
                "verifier": ("work-idle", "comm-idle", None),
            }
        elif step == 1:
            return {
                "proposer": ("work-idle", "comm-idle", None),
                "critic": ("work-idle", "comm-idle", None),
                "verifier": ("quick-verify", "submit-score", None),
            }
        else:
            return {
                "proposer": ("work-idle", "comm-idle", None),
                "critic": ("work-idle", "comm-idle", None),
                "verifier": ("work-idle", "comm-idle", None),
            }
