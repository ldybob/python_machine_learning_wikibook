# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import svm, metrics, model_selection
import random, re

csv = pd.read_csv("./data/iris.csv")

data = csv[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
label = csv["Name"]

clf = svm.SVC()
scores = model_selection.cross_val_score(clf, data, label, cv=5)
print("score = ", scores)
print("avg score = ", scores.mean())