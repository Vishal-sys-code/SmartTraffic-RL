from .env import UrbanTrafficEnv
from gymnasium.envs.registration import register

register(
    id='UrbanTraffic-v0',
    entry_point='smart_traffic_env.env:UrbanTrafficEnv',
)