import os

import pytest
from agent_f import QAgent

from models.rl_m import BleuPrint, ObsidianRobotCost, GeoRobotCost

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

one_blue_print = BleuPrint(blue_print_number=1,
                           ore_robot_cost=1,
                           clay_robot_cost=2,
                           obsidian_robot_cost=ObsidianRobotCost(ore=1, clay=1),
                           geo_robot_cost=GeoRobotCost(ore=2, obsidian=2))

# state => # ore, clay, obsidian, geo, ore_robots, clay_robots, obsidian_robots, geo_robots
test_data = [
    ((one_blue_print, (1, 0, 0, 0, 0, 0, 0, 0)), [0, 1]),
    ((one_blue_print, (2, 0, 0, 0, 0, 0, 0, 0)), [0, 1, 2]),
    ((one_blue_print, (2, 2, 0, 0, 0, 0, 0, 0)), [0, 1, 2, 3]),
    ((one_blue_print, (2, 2, 2, 0, 0, 0, 0, 0)), [0, 1, 2, 3, 4]),
]


@pytest.mark.parametrize("inputs, outputs", test_data)
def test_agent_get_possible_actions(inputs, outputs):
    blue_print, state = inputs
    agent = QAgent(blue_print=blue_print)
    action_possible_result = agent._get_possible_actions(state)
    assert action_possible_result == outputs


test_data = [
    ((one_blue_print, (1, 0, 0, 0, 0, 0, 0, 0), [0, 1, 2], [0, 1, 2, 0, 5]), 2),
]


@pytest.mark.parametrize("inputs, outputs", test_data)
def test_agent__get_max_action_within_a_range_of_allowed_action(inputs, outputs):
    blue_print, state, action_allowed, action_value_state = inputs
    agent = QAgent(blue_print=blue_print)
    agent.Q_table[state] = action_value_state
    result = agent._get_max_action_within_a_range_of_allowed_action(state, allowed_actions=action_allowed)
    assert result == outputs


test_data = [
    ((one_blue_print, (2, 0, 0, 0, 0, 0, 0, 0), [0, 0, 1, 0, 0]), 2),
    ((one_blue_print, (2, 0, 0, 0, 0, 0, 0, 0), [0, 1, 0, 0, 0]), 1),
    ((one_blue_print, (2, 0, 0, 0, 0, 0, 0, 0), [0, 1, 2, 0, 5]), 2),
]


@pytest.mark.parametrize("inputs, outputs", test_data)
def test_agent_policy(inputs, outputs):
    blue_print, state, action_value_state = inputs
    agent = QAgent(blue_print=blue_print, eps=0)

    agent.Q_table[state] = action_value_state
    action_result = agent.policy(state)
    assert action_result == outputs
