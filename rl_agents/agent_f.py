import os
from typing import Tuple, List

import numpy as np
from loguru import logger

from config import action_mapping
from models.rl_m import BleuPrint
from variables import model_folder

# set random seed
np.random.seed(42)


class Agent:
    def __init__(self, blue_print: BleuPrint, load_q_table: bool, eps: float = 0.1, gamma: float = 0.9, alpha: float = 0.2):
        self.blue_print = blue_print
        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha
        if load_q_table:
            q_table_file = os.path.join(model_folder, f"{blue_print.blue_print_number}_Q_table.txt")
            if os.path.exists(q_table_file):
                with open(q_table_file) as f:
                    q_txt = f.read()
                    self.Q_table = eval(q_txt)
                    logger.info(f"Q_table loaded : {q_table_file}")
            else:
                logger.info(f"{q_table_file} not found, starting from scratch")

    def _get_possible_actions(self, state: Tuple[int, int, int, int, int, int, int, int]) -> List[int]:
        ore, clay, obsidian, geo, ore_robots, clay_robots, obsidian_robots, geo_robots = state
        possible_action = [action_mapping["do_nothing"]]
        if ore >= self.blue_print.ore_robot_cost:
            possible_action.append(action_mapping["build_ore_robot"])
        if ore >= self.blue_print.clay_robot_cost:
            possible_action.append(action_mapping["build_clay_robot"])
        if ore >= self.blue_print.obsidian_robot_cost.ore and clay >= self.blue_print.obsidian_robot_cost.clay:
            possible_action.append(action_mapping["build_obsidian_robot"])
        if ore >= self.blue_print.geo_robot_cost.ore and obsidian >= self.blue_print.geo_robot_cost.obsidian:
            possible_action.append(action_mapping["build_geo_robot"])
        return possible_action

    def _get_max_action_within_a_range_of_allowed_action(self, state: Tuple[int, int, int, int, int, int, int, int], allowed_actions: List[int]) -> int:
        """Get the max action within a range of allowed action"""
        max_index = -1
        max_value = -float('inf')

        for action in allowed_actions:
            if self.Q_table[state][action] > max_value:
                max_index = action
                max_value = self.Q_table[state][action]

        if max_index == -1:
            # There are no allowed actions, so return some default value
            return 0
        else:
            return max_index

    def policy(self):
        """need to be implemented"""
        raise NotImplementedError