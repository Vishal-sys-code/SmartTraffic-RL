import gymnasium as gym
import smart_traffic_env

try:
    env = gym.make('UrbanTraffic-v0')
    print("Environment created successfully!")
except Exception as e:
    print(f"Error creating environment: {e}")