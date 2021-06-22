import csv
import datetime
import random

import click

import torch
from torch import nn
from torch.functional import F
from torch.utils.data import Dataset, DataLoader

from fifa_ranking import FIFARanking


class MyModel(nn.Module):
    """ Inputs:
      - 2 ranking (home, away)
      - 3 odds: hone_win, draw, away_win

    Output:
      - total amount of goals
      - delta
    """

    def __init__(self):
        super(MyModel, self).__init__()
        self.seq = nn.Sequential(
            # First hidden layer: 5 inputs 5 outputs
            nn.Linear(5, 5),
            # nn.ReLU(5),
            # Second hidden layer: 5 - 5
            nn.Linear(5, 5),
            # nn.ReLU(5),
            # Output layer: 2 outputs
            nn.Linear(5, 2),
        )

    def forward(self, x):
        x = self.seq(x)
        # We want integer goal amount
        #x = torch.round(x)
        return x


def bet_loss(pred, real):
    """ That's not working as I don't know tensor arithmetic """
    print(pred, real)
    if torch.eq(pred, real):
        # that's a 0
        return F.l1_loss(pred, real, reduction='mean')
    elif pred[0] - pred[1] == real[0] - real[1]:
        # mean, so reduce less when going away
        return F.l1_loss(pred, real, reduction='mean')
    elif (pred[0] > pred[1] and real[0] > real[1]) or (pred[0] < pred[1] and real[0] < real[1]):
        # sum, so reduce more the farther we are
        return F.l1_loss(pred, real, reduction='sum')
    else:
        # MSE: big loss
        return F.mse_loss(pred, real, reduction='sum')


RANKING = FIFARanking()


class EuroDataSet(Dataset):
    def __init__(self, year, train=True):
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
        score = [
            score[0] + score[1],  # sum
            score[0] - score[1]  # delta
        ]
        return torch.tensor(ranking + odds), torch.tensor(score, dtype=torch.float)


LEARNING_RATE = 1e-3

def pred_to_score(pred):
    total, delta = pred
    away = (total - delta) / 2
    home = total - away
    return home, away

def bet_score(expected, actual):
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

@click.group()
def cli():
    pass


@cli.command()
@click.option("--epochs", default=500)
@click.option("--batch-size", default=30)
@click.option("--load", "load_path")
def train(epochs, batch_size, load_path):
    if load_path is None:
        model = MyModel()
    else:
        model = torch.load(load_path)
    print(model)

    loss = bet_loss
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(epochs):
        year = random.choice([2004, 2008, 2012, 2016, 2020])
        totloss = 0
        # Use it in chronological order
        for input, output in DataLoader(EuroDataSet(year), batch_size=batch_size, shuffle=True):
            # predict a round
            pred = model(input)
            #print(f"Predicted {pred[0]}:{pred[1]} for {output[0]}:{output[1]}")
            # calculate the loss
            l = loss(pred, output)
            #print(f"loss: {l.item()}")
            totloss += l.item()

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f'epoch: {epoch}, err: {totloss: 3f}')
            print(dict(list(model.named_parameters())))

        if totloss < 0.0005:
            break

    print(dict(list(model.named_parameters())))
    torch.save(model, "torch_final.pth")


@cli.command()
@click.argument("model_path")
@click.argument("r_home", type=click.INT)
@click.argument("r_away", type=click.INT)
@click.argument("o_home", type=click.FLOAT)
@click.argument("o_draw", type=click.FLOAT)
@click.argument("o_away", type=click.FLOAT)
def test(model_path, r_home, r_away, o_home, o_draw, o_away):
    model = torch.load(model_path)
    input = torch.tensor([r_home, r_away, o_home, o_draw, o_away])
    pred = model(input)
    total, delta = pred
    away = total - delta
    home = total - away
    print(f"The model predicted {home:.0f}:{away:.0f}")

@cli.command()
@click.argument("model_path")
@click.argument("year")
def evaluate(model_path, year):
    points = 0
    model = torch.load(model_path)
    data_set = EuroDataSet(year, False)
    for _in, out in data_set:
        pred = model(_in)
        expected = pred_to_score(pred)
        actual = pred_to_score(out)
        point = bet_score(expected, actual)
        print(f"Predicted {expected[0]:.0f}:{expected[1]:.0f} for {actual[0]:.0f}:{actual[1]:.0f} => +{point}")
        points += point
    print(f"Final points: {points}")


if __name__ == "__main__":
    cli()
