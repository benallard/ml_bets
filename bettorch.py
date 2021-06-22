import csv
import random

import click

import torch
from torch import nn
from torch.functional import F
from torch.utils.data import Dataset, DataLoader

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.seq = nn.Sequential(
            # First hidden layer: 3 inputs 5 outputs
            nn.Linear(3, 5),
            #nn.ReLU(5),
            # Second hidden layer: 5 - 5
            nn.Linear(5, 5),
            #nn.ReLU(5),
            # Output layer: 2 outputs
            nn.Linear(5, 2),
            # Make it positive with a ReLU
            nn.ReLU(2),
        )

    def forward(self, x):
        x = self.seq(x)
        # We want integer goal amount
        x = torch.round(x)
        return x

def bet_loss(pred, real):
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

class EuroDataSet(Dataset):
    def __init__(self, year):
        with open(f"{year}.csv") as f:
            self.data = list(csv.DictReader(f))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        input = [float(o) for o in (datum['mean_odd_home'], datum['mean_odd_draw'], datum['mean_odd_away'])]
        score = datum['score'].split('\xa0')[0]
        output = [int(i) for i in score.split(':')]
        return torch.tensor(input), torch.tensor(output, dtype=torch.float)

LEARNING_RATE=1e5

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
    loss = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(epochs):
        year = random.choice([2004, 2008, 2012, 2016])
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

if __name__ == "__main__":
    cli()