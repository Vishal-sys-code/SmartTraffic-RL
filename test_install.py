import gymnasium as gym
import smart_traffic_env  # noqa: F401

try:
    env = gym.make('UrbanTraffic-v0')
    print("Environment created successfully!")
except Exception as e:
    print(f"Error creating environment: {e}")
