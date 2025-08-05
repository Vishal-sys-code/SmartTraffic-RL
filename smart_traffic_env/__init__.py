from gymnasium.envs.registration import register

# Expose the environment class
from .env import UrbanTrafficEnv

# Register the environment
register(
    id='SmartTraffic-v0',
    entry_point='smart_traffic_env.env:UrbanTrafficEnv',
)