import numpy as np


class DemandGenerator:
    """
    Generates a synthetic demand trajectory for the traffic network.
    The demand is modeled as a noisy sinusoid.
    """

    def __init__(
        self,
        num_steps: int,
        num_lanes: int,
        rng: np.random.RandomState,
        base_demand: float = 100.0,
        period: int = 30,
        amplitude: float = 40.0,
        noise_std: float = 10.0,
    ):
        self.num_steps = num_steps
        self.num_lanes = num_lanes
        self.rng = rng
        self.base_demand = base_demand
        self.period = period
        self.amplitude = amplitude
        self.noise_std = noise_std

    def generate(self) -> np.ndarray:
        """
        Generates the demand trajectory.

        Returns:
            A numpy array of shape (num_steps, num_lanes) representing
            the number of vehicles arriving per second.
        """
        demand = np.zeros((self.num_steps, self.num_lanes))
        time = np.arange(self.num_steps)

        for lane in range(self.num_lanes):
            # Introduce some phase shift for each lane for variety
            phase_shift = self.rng.uniform(0, self.period)

            # Sinusoidal pattern
            sinusoid = self.base_demand + self.amplitude * np.sin(
                2 * np.pi * (time - phase_shift) / self.period
            )

            # Add random noise
            noise = self.rng.normal(0, self.noise_std, self.num_steps)

            # Combine and ensure non-negative demand
            lane_demand = np.maximum(0, sinusoid + noise)

            # Convert from vehicles per hour to vehicles per second (assuming control interval is 1 hr for this)
            # The env will multiply by control_interval later.
            demand[:, lane] = lane_demand / 3600.0

        return demand.astype(np.float32)
