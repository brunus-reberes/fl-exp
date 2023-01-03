import pandas as pd
from sklearn.model_selection import train_test_split

def load_datasets():
    df = pd.read_csv('src/datasets/iris_csv.csv')
    train, test = train_test_split(df)
    train = train.iloc[:, :-1].values, train.iloc[:, [-1]].values
    test = test.iloc[:, :-1].values, test.iloc[:, [-1]].values
    return train, test