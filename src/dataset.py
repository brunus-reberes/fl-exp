import json
import numpy as np
from sklearn.model_selection import train_test_split

def load_iris(n_clients: int):
    with open("src/datasets/iris.json", 'r') as file:
        data = json.loads(file.read())
    train = data[:110]
    test = data[110:]

    return train, val, test

print(len(load_iris(0)[2]))