import numpy as np
from tqdm import tqdm

from rl_agents.agent_f import QAgent
from rl_env.basic_env import MiningEnv


def q_learning_training(nb_episodes: int, agent: QAgent, env: MiningEnv):
    for _ in tqdm(range(nb_episodes)):
        terminated = False
        state = env.initial_state
        minute = 1
        while not terminated:
            action = agent.policy(state)
            next_state, reward, terminated, truncated, information = env.step(action=action, current_state=state, minute=minute)
            if next_state not in agent.Q_table:
                agent.Q_table[next_state] = agent.action_space.copy()
            agent.Q_table[state][action] = agent.Q_table[state][action] + agent.alpha * (
                    reward + agent.gamma * np.max(agent.Q_table[next_state]) - agent.Q_table[state][action]
            )
            state = next_state
            minute += 1
            # if last episode then save the data
    return agent
