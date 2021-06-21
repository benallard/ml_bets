import csv
import random

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # First hidden layer: 3 inputs 5 outputs
        self.hidden1 = nn.Linear(3, 5)
        # Second hidden layer: 5 - 5
        self.hidden2 = nn.Linear(5, 5)
        # Output layer: 2 outputs
        self.output = nn.Linear(5, 2)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.output(x)
        return torch.round(x)

LEARNING_RATE=0.05
N_EPOCHS = 1000
BATCH_SIZE = 30

model = MyModel()

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

for epoch in range(N_EPOCHS):
    year = random.choice([2004, 2008, 2012, 2016])
    with open(f"{year}.csv") as f:
        data = list(csv.DictReader(f))
    totloss = 0
    # Use it in chronological order
    for batch in range(BATCH_SIZE):
        # take a random sample
        datum = random.choice(data)
        # extract the data
        input = [float(o) for o in (datum['mean_odd_home'], datum['mean_odd_draw'], datum['mean_odd_away'])]
        score = datum['score'].split('\xa0')[0]
        output = [int(i) for i in score.split(':')]

        # Reset
        optimizer.zero_grad()

        # predict a round
        pred = model(torch.tensor(input, dtype=torch.float))
        #print(f"Predicted {pred[0]}:{pred[1]} for {output[0]}:{output[1]}")
        # calculate the loss
        l = loss(pred, torch.tensor(output, dtype=torch.float))
        #print(f"loss: {l}")
        totloss += l.item()

        l.backward()

        optimizer.step()



    if epoch % 10 == 0:
        print(f'epoch: {epoch}, err: {totloss: 3f}')
        print(dict(list(model.named_parameters())))


    if totloss < 0.0005:
        break


print(dict(list(model.named_parameters())))