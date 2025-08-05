import pytest
import numpy as np
from smart_traffic_env.env import UrbanTrafficEnv


@pytest.fixture
def env():
    """Pytest fixture to create a default UrbanTrafficEnv instance."""
    return UrbanTrafficEnv(
        num_intersections=2,
        lanes_per_intersection=2,
        episode_length=10,
        seed=42
    )


def test_reset(env):
    """Verify reset() returns an observation of the correct shape and content."""
    obs, info = env.reset()

    # Check observation shape
    N = env.num_intersections
    M = env.num_lanes
    assert obs.shape == (2 * M + N,), "Observation shape is incorrect"

    # Check that initial queues are all zero
    initial_queues = obs[:M]
    assert np.all(initial_queues == 0), "Initial queues should be zero"

    # Check info dictionary
    assert "queues" in info
    assert "greens" in info
    assert np.all(info["queues"] == 0)


def test_step_zero_action_zero_demand(env):
    """Check that a zero-action step on zero demand leaves queues at zero."""
    env.reset()
    
    # Override demand trajectory to be all zeros
    env.demand_trajectory = np.zeros_like(env.demand_trajectory)
    
    zero_action = np.zeros(env.num_intersections)
    obs, reward, _, _, _ = env.step(zero_action)

    queues = obs[:env.num_lanes]
    assert np.all(queues == 0), "Queues should remain zero with no demand"
    assert reward == 0.0, "Reward should be zero for zero queues"


def test_reward_is_negative_with_queues(env):
    """Confirm that reward is negative when queues exist."""
    env.reset()
    
    # Manually set queues to a large value that won't be cleared in one step
    env.queues = np.full(env.num_lanes, 100, dtype=np.float32)
    
    # Take a step with zero demand to isolate reward calculation
    env.demand_trajectory[0] = np.zeros(env.num_lanes)
    
    zero_action = np.zeros(env.num_intersections)
    _, reward, _, _, _ = env.step(zero_action)
    
    # The queues will be reduced by the service rate, but should still be positive
    assert np.all(env.queues > 0)
    assert reward < 0, "Reward should be negative when queues are non-zero"


def test_step_dynamics(env):
    """Check the full state transition with known inputs."""
    obs, _ = env.reset(seed=42)

    # Define a known demand for the first step (vehicles per control interval)
    demand_per_interval = np.array([10, 10, 5, 5], dtype=np.float32)
    env.demand_trajectory[0] = demand_per_interval / env.control_interval # convert to veh/s

    # Define a known action
    action = np.array([2.0, -1.0], dtype=np.float32)

    # --- Theoretical calculation ---
    # 1. Green times
    initial_greens = np.full(env.num_intersections, env.base_green)
    expected_greens = np.clip(initial_greens + action, 10.0, 60.0)

    # 2. Service rate
    from smart_traffic_env.utils import compute_service_rate, DEFAULT_ALL_RED, DEFAULT_SAT_FLOW
    expected_service = compute_service_rate(
        expected_greens,
        env.lanes_per_intersection,
        DEFAULT_ALL_RED,
        DEFAULT_SAT_FLOW,
    )

    # 3. Queue update
    expected_queues = np.maximum(
        0,
        0 + (env.demand_trajectory[0] * env.control_interval) - (expected_service * env.control_interval)
    )
    
    # --- Actual step ---
    next_obs, _, _, _, _ = env.step(action)

    # --- Assertions ---
    actual_queues = next_obs[:env.num_lanes]
    actual_greens = env.greens
    
    assert np.allclose(actual_greens, expected_greens), "Green times did not update correctly"
    assert np.allclose(actual_queues, expected_queues, atol=1e-5), "Queue dynamics are incorrect"