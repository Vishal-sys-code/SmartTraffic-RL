#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_average_delay(log_files: list, labels: list, output_path: str):
    """
    Plots the average delay per episode from log files.

    Args:
        log_files: A list of paths to the log files.
        labels: A list of labels for the plot legend.
        output_path: The path to save the plot image.
    """
    plt.figure(figsize=(12, 8))

    for log_file, label in zip(log_files, labels):
        with open(log_file, 'r') as f:
            data = json.load(f)

        episode_avg_delays = []
        for episode in data:
            # Sum of queues at each timestep
            total_queues_per_step = [np.sum(q) for q in episode["queues"]]
            # Average delay over the episode
            avg_delay = np.mean(total_queues_per_step)
            episode_avg_delays.append(avg_delay)

        plt.plot(range(1, len(episode_avg_delays) + 1), episode_avg_delays, marker='o', linestyle='-', label=label)

    plt.title("Average Vehicle Delay per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Average Delay (Total Queue Length)")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot evaluation results.")
    parser.add_argument("--log_files", nargs='+', required=True, help="List of log files to plot.")
    parser.add_argument("--labels", nargs='+', required=True, help="List of labels for the plot.")
    parser.add_argument("--output_path", type=str, default="average_delay.png", help="Path to save the plot.")
    args = parser.parse_args()

    plot_average_delay(args.log_files, args.labels, args.output_path)