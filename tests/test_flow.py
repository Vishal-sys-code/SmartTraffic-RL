import numpy as np
from smart_traffic_env.flow_model import compute_service_rate, DEFAULT_ALL_RED, DEFAULT_SAT_FLOW


def test_service_rate_conservation():
    """
    Validate that the total service provided across all lanes in a cycle
    equals the theoretical maximum capacity.
    """
    # Test case parameters
    num_intersections = 4
    lanes_per_intersection = 2
    greens = np.array([20, 30, 25, 35], dtype=np.float32)  # Green times for each intersection

    # --- Theoretical Calculation ---
    # Cycle length
    cycle_length = np.sum(greens) + num_intersections * DEFAULT_ALL_RED

    # Total flow capacity for the entire system in one cycle
    # is the sum of (green_time * saturation_flow * num_lanes) for each intersection.
    expected_total_flow = np.sum(greens * DEFAULT_SAT_FLOW) * lanes_per_intersection

    # --- Calculation using the function ---
    # Get service rate per lane (in veh/s)
    service_rate_per_lane = compute_service_rate(
        greens=greens,
        num_lanes=lanes_per_intersection,
        all_red_time=DEFAULT_ALL_RED,
        saturation_flow=DEFAULT_SAT_FLOW,
    )

    # Total served vehicles in one cycle
    # (sum of service_rate_per_lane * cycle_length)
    actual_total_flow = np.sum(service_rate_per_lane * cycle_length)

    # --- Assertion ---
    assert np.isclose(
        actual_total_flow, expected_total_flow
    ), "Total flow provided should equal theoretical capacity"
