#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from smart_traffic_env.env import UrbanTrafficEnv
import os


def run_fixed_time_agent(episodes: int = 10):
    """
    Runs a fixed-time agent for a specified number of episodes.

    Args:
        episodes: The number of episodes to run.
    """
    env = UrbanTrafficEnv()
    for i in range(episodes):
        obs, info = env.reset()
        done = False
        total_queue = 0
        steps = 0
        while not done:
            # A zero action corresponds to no change in green times
            action = np.zeros(env.action_space.shape)
            obs, reward, done, _, info = env.step(action)
            total_queue += np.mean(info["queues"])
            steps += 1
        print(f"Episode {i+1}: Average queue length = {total_queue / steps:.2f}")


def evaluate_model(model_path: str, episodes: int = 10):
    """
    Evaluates a trained PPO model.

    Args:
        model_path: Path to the trained model.
        episodes: The number of episodes to run.
    """
    # Create the environment
    env = DummyVecEnv([lambda: UrbanTrafficEnv()])

    # Load the statistics
    stats_path = os.path.join(os.path.dirname(model_path), "vecnormalize.pkl")
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print("Warning: vecnormalize.pkl not found. Running without normalization.")

    # Load the model
    model = PPO.load(model_path, env=env)

    total_queues = 0
    total_rewards = 0
    total_steps = 0

    for i in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_queues += np.mean(info["queues"])
            total_rewards += reward
            total_steps += 1

    print("--- PPO Agent Evaluation ---")
    print(f"Average queue length: {total_queues / total_steps:.2f}")
    print(f"Average reward: {total_rewards / total_steps:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes for evaluation.")
    args = parser.parse_args()

    evaluate_model(args.model_path, args.episodes)

    print("\n--- Fixed-Time Agent Evaluation ---")
    run_fixed_time_agent(episodes=args.episodes)
