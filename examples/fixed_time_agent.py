import numpy as np
from smart_traffic_env.env import UrbanTrafficEnv

def fixed_time_controller(env, green_time):
    obs, info = env.reset()
    total_queue = 0
    for _ in range(env.episode_length):
        action = np.zeros(env.num_intersections)  # delta = 0 → keeps base_green
        obs, reward, done, truncated, info = env.step(action)
        total_queue += np.mean(env.queues)
        if done or truncated:
            break
    return total_queue / env.episode_length

env = UrbanTrafficEnv()
print("Fixed-time avg queue:", fixed_time_controller(env, env.base_green))
