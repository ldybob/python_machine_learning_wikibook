# -*- coding: utf-8 -*-

import urllib.request as req

local = "./data/mushroom.csv"
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
req.urlretrieve(url,local)
print("ok")