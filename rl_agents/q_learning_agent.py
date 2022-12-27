from typing import Tuple

import numpy as np

from models.rl_m import BleuPrint
from rl_agents.agent_f import Agent


class QLearningAgent(Agent):
    def __init__(self, blue_print: BleuPrint, load_q_table: bool, eps: float = 0.1, gamma: float = 0.9, alpha: float = 0.2):
        super().__init__(blue_print, load_q_table, eps, gamma, alpha)
        self.action_space = [0, 0, 0, 0, 0]
        self.Q_table = {}


    def policy(self, state: Tuple[int, int, int, int, int, int, int, int], eval=False) -> int:
        """Epsilon greedy policy"""
        if state not in self.Q_table:
            self.Q_table[state] = self.action_space.copy()
        possible_actions = self._get_possible_actions(state)
        if (np.random.random() < self.eps or state not in self.Q_table) and not eval:
            return np.random.choice(possible_actions)
        else:
            action = self._get_max_action_within_a_range_of_allowed_action(state, possible_actions)
            return action


    def update_q_table(self, state: Tuple[int, int, int, int, int, int, int, int], action: int, reward: float,
                       next_state: Tuple[int, int, int, int, int, int, int, int]) -> None:
        if next_state not in self.Q_table:
            self.Q_table[next_state] = self.action_space.copy()
        self.Q_table[state][action] = self.Q_table[state][action] + self.alpha * (
                reward + self.gamma * np.max(self.Q_table[next_state]) - self.Q_table[state][action]
        )
