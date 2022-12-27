import numpy as np
from tqdm import tqdm

from rl_agents.q_learning_agent import QLearningAgent
from rl_env.basic_env import MiningEnv


def training(nb_episodes: int, agent: QLearningAgent, env: MiningEnv):
    for _ in tqdm(range(nb_episodes)):
        terminated = False
        state = env.initial_state
        minute = 1
        while not terminated:
            action = agent.policy(state)
            next_state, reward, terminated, truncated, information = env.step(action=action, current_state=state, minute=minute)
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            minute += 1
            # if last episode then save the data
    return agent
