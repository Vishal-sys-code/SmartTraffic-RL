import gymnasium as gym
from gymnasium.envs.registration import register

print("--- Attempting to register environment manually ---")
try:
    register(
        id='SmartTraffic-v0',
        entry_point='smart_traffic_env.env:UrbanTrafficEnv',
    )
    print("Manual registration successful.")
except Exception as e:
    print(f"Manual registration failed: {e}")


print("\n--- Attempting to make 'SmartTraffic-v0' after manual registration ---")
try:
    env = gym.make('SmartTraffic-v0')
    print("Success! Environment created.")
    print(env)
except Exception as e:
    print(f"Failed with error: {e}")
