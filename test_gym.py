import gym
from gym.wrappers.flatten_observation import FlattenObservation

import gym_euro


from stable_baselines3.common.env_checker import check_env

env = FlattenObservation(gym.make('soccer-bet-v0'))

check_env(env)