import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional

from .demand import DemandGenerator
from .flow_model import (
    compute_reward,
    compute_service_rate,
    G_MAX,
    G_MIN,
    DEFAULT_ALL_RED,
    DEFAULT_SAT_FLOW,
)


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
        demand_profile: Optional[np.ndarray] = None,
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

        self.num_lanes = self.num_intersections * self.lanes_per_intersection

        self.demand_gen = DemandGenerator(
            num_steps=episode_length,
            num_lanes=self.num_lanes,
            rng=self.rng
        )

        N = self.num_intersections
        M = self.num_lanes

        # Define constants for observation space bounds
        MAX_QUEUE_VEHICLES = 5000.0
        MAX_DEMAND_VEHICLES_PER_HOUR = 500.0

        obs_low = np.concatenate([
            np.zeros(M, dtype=np.float32),  # Queues
            np.zeros(M, dtype=np.float32),  # Demand
            np.full(N, G_MIN, dtype=np.float32)   # Greens
        ])
        obs_high = np.concatenate([
            np.full(M, MAX_QUEUE_VEHICLES, dtype=np.float32),
            np.full(M, MAX_DEMAND_VEHICLES_PER_HOUR, dtype=np.float32),
            np.full(N, G_MAX, dtype=np.float32)
        ])

        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-self.delta_max, high=self.delta_max, shape=(N,), dtype=np.float32
        )

        # Env state
        self.step_count = 0
        self.queues = np.zeros(M, dtype=np.float32)
        self.greens = np.full(N, self.base_green, dtype=np.float32)
        self.demand_trajectory = None

        # For diagnostics
        self.current_reward = 0
        self.current_info = {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Executes one time step within the environment."""
        # 1. Clip action to be within [-delta_max, +delta_max]
        clipped_action = np.clip(action, -self.delta_max, self.delta_max)

        # 2. Update green times
        self.greens = np.clip(
            self.greens + clipped_action, G_MIN, G_MAX
        ).astype(np.float32)

        # 3. Fetch current demand
        arrival = self.demand_trajectory[self.step_count]

        # 4. Compute service rates
        service = compute_service_rate(
            greens=self.greens,
            num_lanes=self.lanes_per_intersection,
            all_red_time=DEFAULT_ALL_RED,
            saturation_flow=DEFAULT_SAT_FLOW,
        )

        # 5. Update queues
        arrival_rate_per_sec = arrival / 3600.0
        self.queues = np.maximum(
            0,
            self.queues
            + (arrival_rate_per_sec * self.control_interval)
            - (service * self.control_interval),
        ).astype(np.float32)

        # 6. Compute reward
        reward = compute_reward(self.queues)
        self.current_reward = reward

        # 7. Increment step count and check for termination
        self.step_count += 1
        done = self.step_count >= self.episode_length

        # 8. Assemble next observation and info
        obs = self._get_obs()
        info = self._get_info()
        self.current_info = info

        return obs, reward, done, False, info  # Gymnasium expects (obs, rew, terminated, truncated, info)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        # Reset the state of the environment to an initial state
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            self.demand_gen.rng = self.rng

        self.step_count = 0
        self.queues = np.zeros(self.num_lanes, dtype=np.float32)
        self.greens = np.full(self.num_intersections, self.base_green, dtype=np.float32)

        if self.demand_profile is not None:
            self.demand_trajectory = self.demand_profile
        else:
            self.demand_trajectory = self.demand_gen.generate()

        # Initial observation
        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def render(self, mode: str = "human"):
        """Prints a one-line summary of the current state."""
        if mode == "human":
            avg_queue = np.mean(self.queues)
            print(
                f"Step: {self.step_count}, "
                f"Avg Queues: {avg_queue:.2f}, "
                f"Greens: {np.array2string(self.greens, precision=1)}"
            )
        else:
            super().render(mode=mode)  # Let gym handle other modes

    def close(self):
        """Clean up any resources."""
        pass

    def _get_obs(self) -> np.ndarray:
        """Constructs the observation vector."""
        # Get next demand slice
        if self.step_count < self.episode_length:
            next_demand = self.demand_trajectory[self.step_count]
        else:
            next_demand = np.zeros(self.num_lanes)

        # Concatenate [queues, next_demand, greens]
        obs = np.concatenate([self.queues, next_demand, self.greens]).astype(np.float32)
        return obs

    def _get_info(self) -> dict:
        """Constructs the info dictionary."""
        return {"queues": self.queues, "greens": self.greens}


class FixedTimeController:
    """
    A simple controller that uses a fixed green time for all intersections.
    """

    def __init__(self, env: UrbanTrafficEnv):
        self.env = env
        self.num_intersections = env.num_intersections

    def run_episode(self) -> float:
        """
        Runs a single episode with a fixed-time policy.

        Returns:
            float: The average queue length over the episode.
        """
        obs, info = self.env.reset()
        total_queue = 0
        done = False
        while not done:
            action = np.zeros(self.num_intersections)  # delta = 0
            obs, reward, done, truncated, info = self.env.step(action)
            total_queue += np.mean(self.env.queues)
            if truncated:
                break
        return total_queue / self.env.step_count


class ActuatedController:
    """
    A simple actuated controller that adjusts green times based on queue lengths.
    """

    def __init__(self, env: UrbanTrafficEnv, gain: float = 0.5):
        self.env = env
        self.num_intersections = env.num_intersections
        self.lanes_per_intersection = env.lanes_per_intersection
        self.gain = gain

    def _compute_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Computes the action based on the current observation (queues).
        """
        queues = obs[:self.env.num_lanes]
        action = np.zeros(self.num_intersections)

        for i in range(self.num_intersections):
            # Assumes a simple 2-lane per intersection setup
            # Lane 2*i vs Lane 2*i+1
            q1 = queues[2 * i]
            q2 = queues[2 * i + 1]
            
            # Action is proportional to the difference in queue lengths
            action[i] = self.gain * (q1 - q2)

        return action

    def run_episode(self) -> float:
        """
        Runs a single episode with the actuated policy.

        Returns:
            float: The average queue length over the episode.
        """
        obs, info = self.env.reset()
        total_queue = 0
        done = False
        while not done:
            action = self._compute_action(obs)
            obs, reward, done, truncated, info = self.env.step(action)
            total_queue += np.mean(self.env.queues)
            if truncated:
                break
        return total_queue / self.env.step_count
