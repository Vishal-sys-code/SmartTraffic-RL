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


def compute_reward(queues: np.ndarray) -> float:
    """
    Computes the reward for the current state. The reward is the negative
    average queue length, which is a common metric in traffic signal control.

    Args:
        queues: Current queue lengths for each lane [M,].

    Returns:
        Scalar reward value.
    """
    # Simplified reward: negative mean of queue lengths.
    # This makes the reward signal more stable and O(1).
    if queues.size == 0:
        return 0.0
    reward = -np.mean(queues)
    return float(reward)