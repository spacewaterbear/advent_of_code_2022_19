import os
from typing import Any

import neptune.new as neptune
from loguru import logger
from tqdm import tqdm

from rl_agents.agent_f import QAgent
from rl_agents.q_learning import q_learning_training
from rl_env.basic_env import MiningEnv
from utils.utils import Utils
from variables import input_data, NEPTUNE_PROJECT, NEPTUNE_API_KEY, model_folder

one_blue_print = input_data[0]
# raw_blue_print = "Blueprint 99: Each ore robot costs 4 ore.  Each clay robot costs 2 ore.  Each obsidian robot costs 3 ore and 14 clay.  Each geode robot " \
#                  "costs 2 ore and 7 obsidian."
raw_blue_print = "Blueprint 98:   Each ore robot costs 2 ore.  Each clay robot costs 3 ore.  Each obsidian robot costs 3 ore and 8 clay.  Each geode robot " \
                 "costs 3 ore and 12 obsidian."


def evaluate_agent(agent: QAgent, env: MiningEnv, fig_title: str, nb_episode: int = 1000) -> tuple[dict[str, float], Any]:
    """Evaluate the agent"""
    nb_geode = 0
    rewards = 0
    final_run_data = []
    for epi in tqdm(range(nb_episode)):
        state = env.initial_state
        minute = 1
        terminated = False
        while not terminated:
            action = agent.policy(state, eval=True)
            next_state, reward, terminated, truncated, information = env.step(action=action, current_state=state, minute=minute)
            state = next_state
            minute += 1
            if epi == nb_episodes - 1:
                final_run_data.append(next_state)
        # get nb geode
        nb_geode += state[3]
        rewards += reward

    avg_nb_geode = nb_geode / nb_episode
    avg_reward = rewards / nb_episode
    logger.info(f"Average geode: {avg_nb_geode}, reward {avg_reward}")

    figure = Utils.plot_data(final_run_data, fig_title)
    return {"avg_nb_geode": avg_nb_geode,
            "avg_nb_rewaard": avg_reward,
            }, figure


def main(nb_episodes: int, params: dict, fig_title: str, agent, env):

    logger.info(f"fig_title: {fig_title}")
    run = neptune.init_run(
        project=NEPTUNE_PROJECT,
        api_token=NEPTUNE_API_KEY,
    )
    params["nb_episodes"] = nb_episodes
    run["parameters"] = params

    agent_trained = q_learning_training(nb_episodes=nb_episodes, agent=agent, env=env)
    # plot the data
    nb_episode_readable = "1M" if nb_episodes >= 1_000_000 else f"{nb_episodes // 1000}K"

    # save Q_table
    q_table_file = os.path.join(model_folder, f"{blue_print_1.blue_print_number}_Q_table.txt")
    with open(q_table_file, "w") as f:
        f.write(str(agent_trained.Q_table))

    fig_title = f"{fig_title}-{nb_episode_readable} epi"
    result, figure = evaluate_agent(agent=agent_trained, env=env, nb_episode=1000, fig_title=fig_title)
    result["note"] = fig_title
    run["metrics"] = result
    run["q_table"].upload(q_table_file)
    run['figure'].upload(figure)
    run.stop()


if __name__ == '__main__':
    nb_episodes = 1000
    eps, alpha, gamma = 0.2, 0.1, 0.9
    eps_range = [0.1, 0.2, 0.3]
    alpha_range = [0.1, 0.2, 0.3]
    gamma_range = [0.99, 0.9, 0.8]
    blue_print_1 = Utils.blue_print_parser(raw_blue_print)
    env = MiningEnv(blue_print=blue_print_1)
    # itere through all the parameters
    for eps in eps_range:
        for alpha in alpha_range:
            for gamma in gamma_range:
                params = {"eps": eps, "alpha": alpha, "gamma": gamma}
                fig_title = f"eps-{eps}-alpha-{alpha}-gamma-{gamma}"
                agent = QAgent(blue_print=blue_print_1, eps=params['eps'],
                               alpha=params['alpha'], gamma=params['gamma'],
                               load_q_table=False)
                main(nb_episodes=nb_episodes, params=params, fig_title=fig_title, agent=agent, env=env)
    logger.info("Done")
