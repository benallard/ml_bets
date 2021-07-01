import csv
import datetime

import torch
from torch.utils.data import Dataset
from fifa_ranking import FIFARanking

YEARS = (2004, 2008, 2012, 2016, 2020)

RANKING = FIFARanking()


class EuroDataSet(Dataset):
    def __init__(self, year, train=True, tensor=True):
        self.output_tensor = tensor
        with open(f"{year}.csv") as f:
            self.data = list(csv.DictReader(f))
        if not train:
            self.data = list(filter(lambda d: 'Qualification' not in d['kind'], self.data))
            self.data.reverse()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        odds = [float(o) for o in (datum['mean_odd_home'],
                                   datum['mean_odd_draw'],
                                   datum['mean_odd_away'])]
        date = datetime.datetime.fromisoformat(datum['date']).date()
        ranking = [RANKING.get_ranking(datum['home'], date),
                   RANKING.get_ranking(datum['away'], date)]
        score = datum['score'].split('\xa0')[0]
        score = [int(i) for i in score.split(':')]
        goals = [
            score[0] + score[1],  # sum
            abs(score[0] - score[1]),  # delta
        ]
        winner = [
            1 if score[0] > score[1] else 0,
            1 if score[0] == score[1] else 0,
            1 if score[0] < score[1] else 0,
        ]
        #print(f"Returning {goals + winner} for {score[0]}:{score[1]}")
        if self.output_tensor:
            return torch.tensor(ranking + odds), torch.tensor(goals + winner, dtype=torch.float)
        else:
            return ranking + odds, goals + winner


def pred_to_score(pred):
    """ pred is a tensor: total goals, delta for winner, home-win, draw, away-win
    >>> pred_to_score((2.41210234, 1.20658326, 0.,1., 0.))
    (1, 1)
    >>> pred_to_score((2.42633852, 1.17897445, 1., 0.,0.))
    (1, 0)
    """
    total, delta = pred[:2]
    if pred[3] == max(pred[2:]):
        #print("predicted draw")
        home = total / 2
        away = home
    else:
        one = (total - delta) / 2
        two = total - one
        if pred[2] == max(pred[2:]):
            #print("predicted home-win")
            home = max(one, two)
            away = min(one, two)
        elif pred[4] == max(pred[2:]):
            #print("predicted away-win")
            home = min(one, two)
            away = max(one, two)
        else:
            raise AssertionError()
    return home, away

def bet_score(expected, actual):
    """expected is a tensor with round values. actual not"""
    expected = round(expected[0].item()), round(expected[1].item())
    if expected[0] == actual[0] and expected[1] == actual[1]:
        reward = 4
    elif expected[0] - expected[1] == actual[0] - actual[1]:
        reward = 3
    elif (expected[0] > expected[1] and actual[0] > actual[1]) or (expected[0] < expected[1] and actual[0] < actual[1]):
        reward = 2
    else:
        reward = 0
    #print(f"{expected}, {actual}, {reward}")
    return reward
