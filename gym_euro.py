import csv
import random

import numpy as np

import gym
from gym import spaces
from gym.envs.registration import register

import iso3166

class SoccerBetEnv(gym.Env):
    metadata = {
        "render.modes": ["terminal"]
    }

    def __init__(self, same_year=None, use_max = False):
        self.same_year = same_year
        self.use_max = use_max
        self.action_space = spaces.Box(low=0, high=10, shape=(2,), dtype=np.int)
        self.observation_space = spaces.Box(low=0, high=500, shape=(3,), dtype=np.float16)

        self.seed()
        self.reset()

    def reset(self):
        self.turns = 0
        self.points = 0

        if self.same_year is None:
            year = random.choice([2004, 2008, 2012, 2016])
            self.year = year
        else:
            self.year = self.same_year
        with open(f"{self.year}.csv") as f:
            self.data = list(csv.DictReader(f))

        return self._observation

    def step(self, action):
        info = {'year': self.year, 'turn': self.turns, 'points': self.points, 'data': self.data[self.turns]}
        home_score, away_score = (round(s) for s in action)
        score = self.data[self.turns]['score'].split('\xa0')[0]
        score = [int(i) for i in score.split(':')]
        if score[0] == home_score and score[1] == away_score:
            reward = 4
        elif score[0] - score[1] == home_score - away_score:
            reward = 3
        elif (score[0] > score[1] and home_score > away_score) or (score[0] < score[1] and home_score < away_score):
            reward = 2
        else:
            reward = 0

        print(f"Predicting {home_score}:{away_score} for a score of {score} => +{reward}")
        self.points += reward

        obs = self._observation
        self.turns += 1
        return obs, reward, self.done, info

    @property
    def done(self):
        return self.turns >= len(self.data)

    @property
    def _observation(self):
        data = self.data[self.turns ]
        if self.use_max:
            odds = (data['max_odd_home'], data['max_odd_draw'], data['max_odd_away'])
        else:
            odds = (data['mean_odd_home'], data['mean_odd_draw'], data['mean_odd_away'])
        #return country_nr(data['home']), country_nr(data['away']), [float(odd) for odd in odds]
        return np.array([float(odd) for odd in odds])

def country_nr(name):
    name = {
        'Czech Republic': 'Czechia',
        'Russia': 'Russian Federation',
        'Wales': ''
    }.get(name, name)
    country = iso3166.countries.get(name)
    return int(country.numeric)

register(
    id='soccer-bet-v0',
    entry_point = SoccerBetEnv
)