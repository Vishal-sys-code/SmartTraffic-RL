#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from smart_traffic_env.env import UrbanTrafficEnv


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


if __name__ == "__main__":
    run_fixed_time_agent()
