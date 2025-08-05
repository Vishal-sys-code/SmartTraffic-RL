from setuptools import setup

setup(name='smart_traffic_env',
      version='0.0.1',
      install_requires=['gymnasium'],
      entry_points={
          'gymnasium.envs': [
              'UrbanTraffic-v0 = smart_traffic_env.env:UrbanTrafficEnv',
          ]
      }
)
