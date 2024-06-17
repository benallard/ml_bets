import random

import click

import torch
from torch import nn
from torch.functional import F
from torch.utils.data import DataLoader

from betmanual import ManualDrawModel, ManualRankingModel, ManualOddModel, PredictorModel

import dataset
from dataset import EuroDataSet, pred_to_score, bet_score


class MyModel(nn.Module):
    """ Inputs:
      - 2 ranking (home, away)
      - 3 odds: hone_win, draw, away_win

    Output:
      - total amount of goals
      - delta (absolute)
      - winner (home, draw, away)
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
            #nn.Linear(5, 2),
        )
        self.output = nn.Linear(5, 2)
        self.categories = nn.Sequential(
            nn.Linear(5, 3),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        x = self.seq(x)
        # We want integer goal amount
        #x = torch.round(x)
        return torch.cat([
            self.output(x),
            torch.softmax(self.categories(x), 0)
        ], dim=-1)


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

LEARNING_RATE = 1e-3

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
        year = random.choice(dataset.YEARS)
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
def predict(model_path, r_home, r_away, o_home, o_draw, o_away):
    model = torch.load(model_path)
    input = torch.tensor([r_home, r_away, o_home, o_draw, o_away])
    pred = model(input)
    total, delta = pred
    away = total - delta
    home = total - away
    print(f"The model predicted {home:.0f}:{away:.0f}")

@cli.command()
@click.argument("year")
@click.option("--model")
def evaluate(year, model):
    points = 0
    model = {
        'odds': ManualOddModel,
        'ranking': ManualRankingModel,
        'draw': ManualDrawModel,
        'calc': PredictorModel,
    }.get(model, lambda : torch.load(model))()
    data_set = EuroDataSet(year, False)
    for _in, out in data_set:
        pred = model(_in)
        #print (pred)
        expected = pred_to_score(pred)
        actual = pred_to_score(out)
        point = bet_score(expected, actual)
        print(f"Predicted {expected[0]:.0f}:{expected[1]:.0f} for {actual[0]:.0f}:{actual[1]:.0f} => +{point}")
        points += point
    print(f"Final points: {points}")


if __name__ == "__main__":
    cli()
