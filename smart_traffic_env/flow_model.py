import numpy as np

# Default physical parameters
DEFAULT_ALL_RED = 5.0  # seconds
DEFAULT_SAT_FLOW = 2400.0 / 3600.0  # veh/s
G_MIN = 10.0  # seconds
G_MAX = 60.0  # seconds
IMBALANCE_KAPPA = 0.1  # Coefficient for queue imbalance penalty


def compute_service_rate(
    greens: np.ndarray,
    num_lanes: int,
    all_red_time: float,
    saturation_flow: float,
) -> np.ndarray:
    """
    Computes the service rate for each lane based on green times.

    Args:
        greens: Current green times for each intersection [N,].
        num_lanes: Number of lanes per intersection.
        all_red_time: All-red time in seconds.
        saturation_flow: Saturation flow rate in veh/s.

    Returns:
        Service rate for each lane [M,].
    """
    num_intersections = len(greens)
    cycle_length = np.sum(greens) + num_intersections * all_red_time

    if cycle_length <= 0:
        return np.zeros(num_intersections * num_lanes)

    # Service rate for each *intersection*
    intersection_service_rate = (greens / cycle_length) * saturation_flow

    # Expand to all lanes
    service_rate_per_lane = np.repeat(intersection_service_rate, num_lanes)

    return service_rate_per_lane


def compute_reward(queues: np.ndarray, imbalance_kappa: float) -> float:
    """
    Computes the reward for the current state.

    Args:
        queues: Current queue lengths for each lane [M,].
        imbalance_kappa: Weight for the queue imbalance penalty.

    Returns:
        Scalar reward value.
    """
    # 1. Total wait penalty (sum of all queues)
    wait_penalty = -np.sum(queues)

    # 2. Imbalance penalty
    if len(queues) > 0:
        mean_queue = np.mean(queues)
        imbalance_penalty = -imbalance_kappa * np.sum(np.abs(queues - mean_queue))
    else:
        imbalance_penalty = 0

    reward = wait_penalty + imbalance_penalty
    return float(reward)
