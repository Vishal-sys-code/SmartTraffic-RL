#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from smart_traffic_env.env import UrbanTrafficEnv


def evaluate_model(model_path: str = "ppo_urbantraffic.zip", episodes: int = 1):
    """
    Evaluates a trained model.

    Args:
        model_path: The path to the trained model.
        episodes: The number of episodes to run.
    """
    env = UrbanTrafficEnv()
    model = PPO.load(model_path)

    for i in range(episodes):
        obs, info = env.reset()
        done = False
        queues = []
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            queues.append(np.mean(info["queues"]))
        
        plt.figure(figsize=(10, 5))
        plt.plot(queues)
        plt.xlabel("Time (steps)")
        plt.ylabel("Average Queue Length")
        plt.title(f"Evaluation of PPO Model (Episode {i+1})")
        plt.grid(True)
        plt.savefig(f"evaluation_episode_{i+1}.png")
        print(f"Evaluation plot saved to evaluation_episode_{i+1}.png")


if __name__ == "__main__":
    evaluate_model()
