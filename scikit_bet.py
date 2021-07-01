import numpy as np
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import dataset

# Get the data ready
input = []
outputreg = []
outputclass = []
for year in dataset.YEARS:
    data = dataset.EuroDataSet(year, tensor=False)
    for x, y in data:
        input.append(x)
        outputreg.append(y[:2])
        outputclass.append(y[2:].index(1))

#Learn
reg = make_pipeline(StandardScaler(),
                    linear_model.LinearRegression())
reg.fit(input, outputreg)
print("learned reg: ", reg[1].coef_)

clf = make_pipeline(StandardScaler(),
                    linear_model.SGDClassifier(loss="hinge", penalty="l2", max_iter=500))
clf.fit(input, outputclass)
print("learned clf: ", clf[1].coef_)

#predict
points = 0
for x, y in dataset.EuroDataSet(2020, False, False):
    goals = reg.predict([x])
    winner = clf.predict([x])
    expected = np.concatenate((goals[0], [
        int(winner[0] == 0),
        int(winner[0] == 1),
        int(winner[0] == 2)]))
    #print(expected)
    expected = dataset.pred_to_score(expected)
    #print (expected)
    actual = dataset.pred_to_score(y)
    point = dataset.bet_score(expected, actual)
    print(f"Predicted {expected[0]:.0f}:{expected[1]:.0f} for {actual[0]:.0f}:{actual[1]:.0f} => +{point}")
    points += point
print(points)