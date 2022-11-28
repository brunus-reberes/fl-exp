from ellyn import ellyn
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

learner = ellyn(classification=True, 
                class_m4gp=True, 
                prto_arch_on=True,
                selection='lexicase',
                fit_type='F1', # can be 'F1' or 'F1W' (weighted F1)
                verbosity=3,
                popsize=10000,
                g=1000
               )


df = pd.read_csv('src/datasets/iris_csv.csv')

train, test = train_test_split(df)
x_train, y_train = train.iloc[:, :-1], train.iloc[:, [-1]]
x_test, y_test = test.iloc[:, :-1], test.iloc[:, [-1]]
learner.fit(x_train.values, y_train.values)