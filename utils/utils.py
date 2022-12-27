import re
import matplotlib.pyplot as plt

from models.rl_m import BleuPrint, ObsidianRobotCost, GeoRobotCost


class Utils:
    @staticmethod
    def blue_print_parser(blue_print: str) -> BleuPrint:
        """Parse the blue print and return the states config"""
        # extract the number from string 'Blueprint 1: ' with regex
        blue_print_nb = re.findall(r"Blueprint (\d+): ", blue_print)[0]
        ore_robot_cost = re.findall(r"Each ore robot costs (\d+) ore", blue_print)[0]
        clay_robot_cost = re.findall(r"Each clay robot costs (\d+) ore", blue_print)[0]
        ore_obsidian_robot_cost, clay_obsidian_robot_cost = re.findall(
            r"Each obsidian robot costs (\d+) ore and (\d+) clay", blue_print
        )[0]
        ore_geo_robot_cost, obsidian_geo_robot_cost = re.findall(
            r"Each geode robot costs (\d+) ore and (\d+) obsidian", blue_print
        )[0]
        return BleuPrint(
            blue_print_number=blue_print_nb,
            ore_robot_cost=ore_robot_cost,
            clay_robot_cost=clay_robot_cost,
            obsidian_robot_cost=ObsidianRobotCost(
                ore=ore_obsidian_robot_cost, clay=clay_obsidian_robot_cost
            ),
            geo_robot_cost=GeoRobotCost(
                ore=ore_geo_robot_cost, obsidian=obsidian_geo_robot_cost
            ),
        )


    @staticmethod
    def plot_data(final_run_data, param):
        """Plot evolution of data of the final run"""
        x = list(range(1, len(final_run_data) + 1))
        labels = ["ore", "clay", "obsidian", "geode", "ore robots", "clay robots", "obsidian robots", "geode robots"]
        for i, label in enumerate(labels):
            plt.plot(
                x,
                [j[i] for j in final_run_data],
                label=label,
            )
        plt.legend()
        plt.title(param)
        # plt.show()
        # return figure
        return plt.gcf()