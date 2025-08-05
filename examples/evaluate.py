#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from stable_baselines3 import PPO
from smart_traffic_env.env import UrbanTrafficEnv
from smart_traffic_env.wrappers import NormalizeObservation
from examples.fixed_time_agent import run_fixed_time_agent


def evaluate_model(model_path: str, episodes: int = 10):
    """
    Evaluates a trained PPO model.

    Args:
        model_path: Path to the trained model.
        episodes: The number of episodes to run.
    """
    env = UrbanTrafficEnv()
    env = NormalizeObservation(env)
    model = PPO.load(model_path)

    total_queues = 0
    total_rewards = 0
    total_steps = 0

    for i in range(episodes):
        obs, info = env.reset()
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
