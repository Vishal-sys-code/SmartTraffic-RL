import numpy as np
from stable_baselines3 import PPO
from smart_traffic_env.env import UrbanTrafficEnv, FixedTimeController, ActuatedController

def evaluate_ppo_model(model_path: str, env: UrbanTrafficEnv, num_episodes: int = 10) -> float:
    """
    Evaluates a PPO model on the environment.

    Args:
        model_path (str): Path to the trained PPO model.
        env (UrbanTrafficEnv): The environment to evaluate on.
        num_episodes (int): Number of episodes to run for evaluation.

    Returns:
        float: Average queue length over all episodes.
    """
    model = PPO.load(model_path)
    total_queue_steps = 0
    total_steps = 0
    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_queue_steps += np.mean(env.queues)
            total_steps += 1

    return total_queue_steps / total_steps if total_steps > 0 else 0

def evaluate_fixed_time_controller(env: UrbanTrafficEnv, num_episodes: int = 10) -> float:
    """
    Evaluates the FixedTimeController on the environment.

    Args:
        env (UrbanTrafficEnv): The environment to evaluate on.
        num_episodes (int): Number of episodes to run for evaluation.

    Returns:
        float: Average queue length over all episodes.
    """
    controller = FixedTimeController(env)
    total_avg_queue = 0
    for _ in range(num_episodes):
        avg_queue_per_episode = controller.run_episode()
        total_avg_queue += avg_queue_per_episode

    return total_avg_queue / num_episodes if num_episodes > 0 else 0

def evaluate_actuated_controller(env: UrbanTrafficEnv, num_episodes: int = 10) -> float:
    """
    Evaluates the ActuatedController on the environment.

    Args:
        env (UrbanTrafficEnv): The environment to evaluate on.
        num_episodes (int): Number of episodes to run for evaluation.

    Returns:
        float: Average queue length over all episodes.
    """
    controller = ActuatedController(env)
    total_avg_queue = 0
    for _ in range(num_episodes):
        avg_queue_per_episode = controller.run_episode()
        total_avg_queue += avg_queue_per_episode

    return total_avg_queue / num_episodes if num_episodes > 0 else 0


if __name__ == "__main__":
    # Environment setup
    env = UrbanTrafficEnv(episode_length=200) # Use a longer episode for more stable results
    NUM_EPISODES = 20

    # PPO Model Evaluation
    ppo_model_path = "examples/logs/ppo_traffic/best_model.zip"
    avg_queue_ppo = evaluate_ppo_model(ppo_model_path, env, NUM_EPISODES)
    print(f"PPO Agent Average Queue Length: {avg_queue_ppo:.2f}")

    # Fixed-Time Controller Evaluation
    avg_queue_fixed = evaluate_fixed_time_controller(env, NUM_EPISODES)
    print(f"Fixed-Time Controller Average Queue Length: {avg_queue_fixed:.2f}")

    # Actuated Controller Evaluation
    avg_queue_actuated = evaluate_actuated_controller(env, NUM_EPISODES)
    print(f"Actuated Controller Average Queue Length: {avg_queue_actuated:.2f}")