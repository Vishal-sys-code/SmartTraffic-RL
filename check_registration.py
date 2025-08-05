import smart_traffic_env 
import gymnasium as gym
from gymnasium.envs.registration import registry

print("--- Registered Environments ---")
print(sorted(registry.keys()))
print("-----------------------------")

print("\nAttempting to make 'SmartTraffic-v0'...")
try:
    env = gym.make('SmartTraffic-v0')
    print("\nSuccess! Environment created.")
except Exception as e:
    print(f"\nFailed with error: {e}")