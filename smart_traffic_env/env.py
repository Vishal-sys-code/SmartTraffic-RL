import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict

from .flow_model import compute_service_rate
from .demand import DemandGenerator


class UrbanTrafficEnv(gym.Env):
    """Gym-style env for urban traffic signal control."""

    def __init__(self):
        super(UrbanTrafficEnv, self).__init__()

        # Define action and observation spaces
        # They must be gym.spaces objects
        # Example when dealing with discrete actions:
        self.action_space = spaces.Discrete(2)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(1, 1, 3), dtype=np.uint8)

    def step(self, action):
        # Execute one time step within the environment
        pass

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        super().reset(seed=seed)
        pass

    def render(self):
        # Render the environment to the screen
        pass