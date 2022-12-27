from enum import Enum

from pydantic import BaseModel


class States(BaseModel):
    ore: int
    ore_robots: int
    clay_robots: int
    obsidian_robots: int
    geo_robots: int


# @dataclass
class ObsidianRobotCost(BaseModel):
    ore: int
    clay: int


# @dataclass
class GeoRobotCost(BaseModel):
    ore: int
    obsidian: int


# @dataclass
class BleuPrint(BaseModel):
    blue_print_number: int
    ore_robot_cost: int
    clay_robot_cost: int
    obsidian_robot_cost: ObsidianRobotCost
    geo_robot_cost: GeoRobotCost


class RLAlgo(str, Enum):
    q_learning_agent = "q_learning_agent"