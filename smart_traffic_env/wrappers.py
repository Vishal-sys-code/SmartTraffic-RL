import gymnasium as gym
import numpy as np


class NormalizeObservation(gym.ObservationWrapper):
    """
    Normalizes the observation space to the range [-1, 1].
    """
    def __init__(self, env):
        super().__init__(env)
        self.obs_low = self.observation_space.low
        self.obs_high = self.observation_space.high
        self.obs_mean = (self.obs_high + self.obs_low) / 2
        self.obs_range = (self.obs_high - self.obs_low) / 2
        
        # Avoid division by zero
        self.obs_range[self.obs_range == 0] = 1

        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=self.observation_space.shape, dtype=np.float32
        )

    def observation(self, obs):
        return (obs - self.obs_mean) / self.obs_range


class ScaleReward(gym.RewardWrapper):
    """
    Scales the reward by a constant factor.
    """
    def __init__(self, env, scale_factor: float):
        super().__init__(env)
        self.scale_factor = scale_factor

    def reward(self, rew):
        return rew / self.scale_factor
