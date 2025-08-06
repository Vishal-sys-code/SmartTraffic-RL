#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from smart_traffic_env.env import UrbanTrafficEnv


def train_ppo(args):
    """
    Trains a PPO agent on the UrbanTrafficEnv.
    """
    # Create a vectorized environment
    env = make_vec_env(UrbanTrafficEnv, n_envs=args.n_envs)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)


    # Define the PPO model
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        clip_range=0.2,
        gamma=args.gamma,
        ent_coef=0.0,
        vf_coef=0.5,
        verbose=1,
        policy_kwargs={"net_arch": dict(pi=[args.net_arch, args.net_arch], vf=[args.net_arch, args.net_arch])},
        tensorboard_log=f"./ppo_tensorboard/{args.exp_name}",
    )

    # Set up evaluation callback
    eval_env = make_vec_env(lambda: Monitor(UrbanTrafficEnv()), n_envs=1)
    eval_env = VecNormalize(eval_env, training=False, norm_reward=False)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./logs/{args.exp_name}",
        log_path=f"./logs/{args.exp_name}",
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        deterministic=True,
        render=False,
    )

    # Train the model
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback
    )
    model.save(f"ppo_urbantraffic_{args.exp_name}")

    print(f"Training finished and model saved to ppo_urbantraffic_{args.exp_name}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO agent for traffic signal control.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--n_steps", type=int, default=4096, help="Number of steps per rollout.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--net_arch", type=int, default=64, help="Size of the policy network layers.")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments.")
    parser.add_argument("--total_timesteps", type=int, default=200_000, help="Total number of training timesteps.")
    parser.add_argument("--eval_freq", type=int, default=8192, help="Evaluation frequency.")
    parser.add_argument("--exp_name", type=str, default="ppo_traffic", help="Experiment name for logging.")
    args = parser.parse_args()
    train_ppo(args)