"""
reward_functions.py - Multi-Objective Reward Formulations for Intrusion Prevention

Balances security efficacy against operational availability and inspection overhead:
- Heavy penalty for False Positives (dropping benign users degrades SLA).
- Substantial reward for True Positives (mitigating attacks protects infrastructure).
- Proportional rewards for Rate Limiting and penalties for unnecessary deep inspection.
"""

from typing import Dict, Tuple

ACTION_PASS = 0
ACTION_DROP = 1
ACTION_RATE_LIMIT = 2
ACTION_DEEP_INSPECT = 3


class IPSRewardFunction:
    """Configurable reward computer for the IPS Gymnasium Environment."""

    def __init__(
        self,
        reward_tp_drop: float = 10.0,
        reward_tn_pass: float = 2.0,
        penalty_fp_drop: float = -50.0,
        penalty_fn_pass: float = -30.0,
        reward_rate_limit_attack: float = 5.0,
        penalty_rate_limit_benign: float = -10.0,
        cost_deep_inspect: float = -0.5,
    ):
        self.reward_tp_drop = reward_tp_drop
        self.reward_tn_pass = reward_tn_pass
        self.penalty_fp_drop = penalty_fp_drop
        self.penalty_fn_pass = penalty_fn_pass
        self.reward_rate_limit_attack = reward_rate_limit_attack
        self.penalty_rate_limit_benign = penalty_rate_limit_benign
        self.cost_deep_inspect = cost_deep_inspect

    def compute(
        self,
        action: int,
        is_attack: bool,
        threat_class: int = 0,
    ) -> Tuple[float, Dict[str, bool]]:
        """
        Compute scalar reward and classification outcome indicator.
        Returns: (reward, outcome_dict)
        """
        outcome = {
            "tp": False,
            "tn": False,
            "fp": False,
            "fn": False,
            "rate_limited_attack": False,
            "rate_limited_benign": False,
            "inspected": False,
        }

        if action == ACTION_PASS:
            if is_attack:
                reward = self.penalty_fn_pass
                outcome["fn"] = True
            else:
                reward = self.reward_tn_pass
                outcome["tn"] = True

        elif action == ACTION_DROP:
            if is_attack:
                reward = self.reward_tp_drop
                outcome["tp"] = True
            else:
                reward = self.penalty_fp_drop
                outcome["fp"] = True

        elif action == ACTION_RATE_LIMIT:
            if is_attack:
                reward = self.reward_rate_limit_attack
                outcome["rate_limited_attack"] = True
            else:
                reward = self.penalty_rate_limit_benign
                outcome["rate_limited_benign"] = True

        elif action == ACTION_DEEP_INSPECT:
            outcome["inspected"] = True
            # Deep inspection resolves the flow with near-certainty
            base_cost = self.cost_deep_inspect
            if is_attack:
                reward = base_cost + self.reward_tp_drop * 0.8
                outcome["tp"] = True
            else:
                reward = base_cost + self.reward_tn_pass * 0.8
                outcome["tn"] = True

        else:
            reward = -1.0

        return reward, outcome
