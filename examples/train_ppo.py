#!/usr/bin/env python
# -*- coding: utf-8 -*-

from stable_baselines3 import PPO
from smart_traffic_env.env import UrbanTrafficEnv

# Note: With default env parameters, the reward signal is very sparse
# as queues are almost always zero. Consider increasing the demand
# or decreasing the service rate in the environment for more meaningful
# training.
env = UrbanTrafficEnv()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("ppo_urbantraffic")

print("Training finished and model saved to ppo_urbantraffic.zip")
