from typing import Tuple, Any

from config import action_mapping
from models.rl_m import BleuPrint


class MiningEnv:
    def __init__(self, blue_print: BleuPrint, nb_minutes: int = 24):
        """Initialize the environment
        """
        self.minutes_duration = nb_minutes
        self.blue_print = blue_print
        self.initial_state = (0, 0, 0, 0, 1, 0, 0, 0)

    def step(self, action: int, current_state: Tuple[int, int, int, int, int, int, int, int], minute: int) -> tuple[
        tuple[int, int, int, int, int, int, int, int], int, bool, bool, dict[Any, Any]]:
        """Return the next state, the reward and if the episode is done"""
        ore, clay, obsidian, geo, ore_robots, clay_robots, obsidian_robots, geo_robots = current_state
        ore += ore_robots
        clay += clay_robots
        obsidian += obsidian_robots
        geo += geo_robots
        reward = -1
        # if clay > 20:
        #     reward += -5
        if action == action_mapping["do_nothing"]:
            pass
            # if ore > 5:
            #     reward += -1

        elif action == action_mapping["build_ore_robot"]:
            ore_robots += 1
            # reward += 1
            ore -= self.blue_print.ore_robot_cost
        elif action == action_mapping["build_clay_robot"]:
            clay_robots += 1
            # reward += 2
            ore -= self.blue_print.clay_robot_cost
        elif action == action_mapping["build_obsidian_robot"]:
            obsidian_robots += 1
            ore -= self.blue_print.obsidian_robot_cost.ore
            clay -= self.blue_print.obsidian_robot_cost.clay
            # reward += 10
        elif action == action_mapping["build_geo_robot"]:
            geo_robots += 1
            ore -= self.blue_print.geo_robot_cost.ore
            obsidian -= self.blue_print.geo_robot_cost.obsidian
            reward += 50
        # convert state to string
        next_state = (ore, clay, obsidian, geo, ore_robots, clay_robots, obsidian_robots, geo_robots)
        if len(next_state) != 8:
            raise ValueError(
                f"State must be a string of 8 digits {next_state} ({ore=}{clay=}{obsidian=}{geo=}{ore_robots=}{clay_robots=}{obsidian_robots=}{geo_robots=})")
        truncated = False
        if minute == self.minutes_duration:
            terminated = True

        else:
            terminated = False
            # reward = 0

        # reward+=obsidian*1
        reward += geo * 500

        information = {}
        return next_state, reward, terminated, truncated, information
