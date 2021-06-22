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

    def __init__(self, same_year=None, use_max = False, max_turns=50):
        self.same_year = same_year
        self.use_max = use_max
        self.max_turns = max_turns
        self.action_space = spaces.Box(low=0, high=10, shape=(2,), dtype=np.int)
        self.observation_space = spaces.Box(low=0, high=500, shape=(3,), dtype=np.float16)

        self.seed()
        self.reset()

    def reset(self):
        self.turns = 0
        self.points = 0

        if self.same_year is None:
            year = random.choice([2004, 2008, 2012, 2016, 2020])
            self.year = year
        else:
            self.year = self.same_year
        with open(f"{self.year}.csv") as f:
            self.data = list(csv.DictReader(f))
        self.current = random.choice(self.data)

        return self._observation

    def step(self, action):
        info = {'year': self.year, 'turn': self.turns, 'points': self.points, 'current': self.current}
        home_score, away_score = (round(s) for s in action)
        score = self.current['score'].split('\xa0')[0]
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
        self.current = random.choice(self.data)
        return obs, reward, self.done, info

    @property
    def done(self):
        return self.turns >= self.max_turns

    @property
    def _observation(self):
        if self.use_max:
            odds = (self.current['max_odd_home'], self.current['max_odd_draw'], self.current['max_odd_away'])
        else:
            odds = (self.current['mean_odd_home'], self.current['mean_odd_draw'], self.current['mean_odd_away'])
        #return country_nr(self.current['home']), country_nr(self.current['away']), [float(odd) for odd in odds]
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