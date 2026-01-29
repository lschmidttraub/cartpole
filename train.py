import matplotlib.pyplot as plt
import numpy as np
import os
import time

from visualize import visualize_episode
if __name__=="__main__":
    from envs.cartpole_env import CartPoleEnv
    from policies.a2c import A2C
    from policies.ppo import PPO
    # from policies.reinforce import Reinforce
    # from policies.ddpg import DDPG

    env = CartPoleEnv()
    agent = PPO(env)
    episodes = 1000
    episode_returns = []
    pi_losses = []
    v_losses = []
    warmup_episodes = 10
    steps_per_episode = 20
    # for _ in range(warmup_episodes):
    #     episode_return = agent.store_episode(random=True)
    #     episode_returns.append(episode_return)
    for _ in range(episodes):
        episode_return = agent.store_episode()
        episode_returns.append(episode_return)
        pi_loss, v_loss = agent.train()
        pi_losses.append(pi_loss)
        v_losses.append(v_loss)
    # Display the return and both losses on separate plots but in the same window
    fig, axs = plt.subplots(3, 1, figsize=(10, 14))

    # Plot episode returns
    axs[0].plot(np.arange(len(episode_returns)), episode_returns, label="Episode Return", color="tab:green")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Episode Return")
    axs[0].set_title("PPO Episode Returns")
    axs[0].grid(True)
    axs[0].legend()

    # Plot pi losses
    axs[1].plot(np.arange(len(pi_losses)), pi_losses, label="Pi loss", color="tab:blue")
    axs[1].set_xlabel("Training Step")
    axs[1].set_ylabel("Pi Loss")
    axs[1].set_title("PPO Policy Loss")
    axs[1].grid(True)
    axs[1].legend()

    # Plot v losses
    axs[2].plot(np.arange(len(v_losses)), v_losses, label="V loss", color="tab:orange")
    axs[2].set_xlabel("Training Step")
    axs[2].set_ylabel("V Loss")
    axs[2].set_title("PPO V-value Loss")
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.show()

