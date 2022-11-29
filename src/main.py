from collections import OrderedDict
from typing import List, Tuple

import flwr as fl
import numpy as np
import matplotlib.pyplot as plt
from flwr.common import Metrics
from ellyn import ellyn
from sklearn.model_selection import train_test_split
import pandas as pd

CLASSES = ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica')
NUM_CLIENTS = 10

BATCH_SIZE = 32

def load_datasets():
    df = pd.read_csv('src/datasets/iris_csv.csv')
    train, test = train_test_split(df)
    train = train.iloc[:, :-1].values, train.iloc[:, [-1]].values
    test = test.iloc[:, :-1].values, test.iloc[:, [-1]].values
    return train, test

train_data, test_data = load_datasets()

model = ellyn(classification=True, 
                class_m4gp=True, 
                prto_arch_on=True,
                selection='lexicase',
                fit_type='F1', # can be 'F1' or 'F1W' (weighted F1)dition=False,
                stop_condition=True
               )

def train(model, train_data, popsize, gen, verbose=False):
    """Train the M4GP on the training set."""
    model.verbosity = verbose
    model.popsize = popsize
    model.g = gen
    model.fit(train_data[0], train_data[1])


def test(model, test_data):
    """Evaluate the network on the entire test set."""
    model.predict(test_data[0])
    score = model.score(test_data[0], test_data[1])
    return score

train(model, train_data, popsize=10000, gen=1000,verbose=False)
print(test(model, test_data))
