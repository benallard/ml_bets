import random

import click

import torch
from torch import nn
from torch.functional import F
from torch.utils.data import DataLoader

from betmanual import ManualDrawModel, ManualRankingModel, ManualOddModel, PredictorModel

from dataset import EuroDataSet, FullDataSet, YEARS, pred_to_score, bet_score


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
            # Input layer: 5 inputs 5 outputs
            nn.Linear(5, 5),
            nn.ReLU(),
            # Hidden layer: 5 - 5
            nn.Linear(5, 5),
            nn.Sigmoid(),
        )
        self.output = nn.Linear(5, 2)
        self.categories = nn.Sequential(
            nn.Linear(5, 3),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        x = self.seq(x)
        # We want integer goal amount
        # x = torch.round(x)
        return torch.cat([
            self.output(x),
            torch.softmax(self.categories(x), 0)
        ], dim=-1)


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        # Separate the parts of the pred tensor
        scores_pred = pred[:, :2]  # First two elements (scores)
        categories_pred = pred[:, 2:]  # Last three elements (categories)

        # Separate the parts of the target tensor
        scores_target = target[:, :2]  # Assuming the target has the same structure
        categories_target = target[:, 2:].argmax(dim=1)  # Assuming category target is the 3rd element

        # Calculate the losses
        loss_scores = self.mse_loss(scores_pred, scores_target)
        loss_categories = self.ce_loss(categories_pred, categories_target)

        # Combine the losses (you can adjust the weighting if necessary)
        total_loss = loss_scores + 5 * loss_categories
        return total_loss


LEARNING_RATE = 0.005


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

    loss = CombinedLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(epochs):
        year = random.choice(YEARS)
        dataset = EuroDataSet(year)
        dataset = FullDataSet()
        totloss = 0
        # Use it in chronological order
        for input, output in DataLoader(dataset, batch_size=batch_size, shuffle=True):
            # predict a round
            pred = model(input)
            # print(f"Predicted {pred[0]}:{pred[1]} for {output[0]}:{output[1]}")
            # calculate the loss
            l = loss(pred, output)
            # print(f"loss: {l.item()}")
            totloss += l.item()

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f'epoch: {epoch}, err: {totloss / len(dataset): 3f}')
            # print(dict(list(model.named_parameters())))

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
@click.option("--model", default='torch_final.pth')
def evaluate(year, model):
    points = 0
    model = {
        'random': MyModel,
        'odds': ManualOddModel,
        'ranking': ManualRankingModel,
        'draw': ManualDrawModel,
        'calc': PredictorModel,
    }.get(model, lambda: torch.load(model))()
    data_set = EuroDataSet(year, False)
    for _in, out in data_set:
        pred = model(_in)
        # print (pred)
        expected = pred_to_score(pred)
        actual = pred_to_score(out)
        point = bet_score(expected, actual)
        print(f"Predicted {expected[0]:.0f}:{expected[1]:.0f} for {actual[0]:.0f}:{actual[1]:.0f} => +{point}")
        points += point
    print(f"Final points: {points}")


if __name__ == "__main__":
    cli()
