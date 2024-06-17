import numpy as np
from sklearn import linear_model, metrics
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
                    linear_model.SGDClassifier(loss="hinge", penalty="l2"))
clf.fit(input, outputclass)
print("learned clf: ", clf[1].coef_)

print(metrics.classification_report(outputclass, clf.predict(input)))#, labels=["home", "draw", "away"])

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

""" 96 points in 2020:
learned reg:  [[-0.01502271 -0.06723512  0.2218396   0.18949771  0.49426794]
               [ 0.10011868 -0.00591067  0.44753095  0.04130997  0.82847817]]
learned clf:  [[-0.01358447  1.01276075 -1.97585216  1.10208288  1.21926538]
               [ 0.08031429 -0.02955787 -0.55122555 -0.26782454  0.04729079]
               [ 0.114992    0.28764012  2.11578578  0.57384395 -1.53268404]]
"""