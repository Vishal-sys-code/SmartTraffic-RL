import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict

from .flow_model import compute_service_rate
from .demand import DemandGenerator


class UrbanTrafficEnv(gym.Env):
    """Gym-style env for urban traffic signal control."""

    def __init__(
        self,
        num_intersections: int = 4,
        lanes_per_intersection: int = 2,
        base_green: float = 30.0,
        delta_max: float = 5.0,
        control_interval: float = 60.0,
        episode_length: int = 60,
        demand_profile: np.ndarray = None,
        seed: int = None,
    ) -> None:
        super(UrbanTrafficEnv, self).__init__()
        self.num_intersections = num_intersections
        self.lanes_per_intersection = lanes_per_intersection
        self.base_green = base_green
        self.delta_max = delta_max
        self.control_interval = control_interval
        self.episode_length = episode_length
        self.demand_profile = demand_profile
        self.seed = seed

        self.rng = np.random.RandomState(seed)

        self.demand_gen = DemandGenerator(
            num_steps=episode_length,
            num_lanes=num_intersections * lanes_per_intersection,
            rng=self.rng
        )

        N = self.num_intersections
        M = self.num_intersections * self.lanes_per_intersection

        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(2 * M + N,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-self.delta_max, high=self.delta_max, shape=(N,), dtype=np.float32
        )

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