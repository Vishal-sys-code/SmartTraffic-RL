import json
import numpy as np

class MetricsLogger:
    """
    Logs time-series data for an episode, including queues, actions, and rewards.
    """
    def __init__(self):
        self.episode_logs = []
        self._current_episode_queues = []
        self._current_episode_actions = []
        self._current_episode_rewards = []

    def log_step(self, queues, action, reward):
        """
        Records the metrics for a single simulation step.

        Args:
            queues (np.ndarray): The current queue lengths.
            action (np.ndarray): The action taken by the agent.
            reward (float): The reward received.
        """
        self._current_episode_queues.append(queues.tolist())
        self._current_episode_actions.append(action.tolist())
        self._current_episode_rewards.append(reward)

    def end_episode(self):
        """
        Aggregates and stores the logs for the completed episode.
        """
        if not self._current_episode_queues:
            return  # Avoid saving empty episodes

        log_entry = {
            "queues": self._current_episode_queues,
            "actions": self._current_episode_actions,
            "rewards": self._current_episode_rewards,
        }
        self.episode_logs.append(log_entry)

        # Reset for the next episode
        self._current_episode_queues = []
        self._current_episode_actions = []
        self._current_episode_rewards = []

    def save(self, filepath: str):
        """
        Saves the logged data to a JSON file.

        Args:
            filepath: The path to the output file.
        """
        with open(filepath, 'w') as f:
            json.dump(self.episode_logs, f, indent=4)
