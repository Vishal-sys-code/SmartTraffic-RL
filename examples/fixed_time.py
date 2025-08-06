from smart_traffic_env.env import UrbanTrafficEnv, FixedTimeController

# Instantiate the environment
env = UrbanTrafficEnv()

# Instantiate the controller
fixed_time_controller = FixedTimeController(env)

# Run a single episode
avg_queue = fixed_time_controller.run_episode()

print(f"Fixed-time avg queue for one episode: {avg_queue:.2f}")