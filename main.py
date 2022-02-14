import pandas as pd
from numpy import absolute, mean, std
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.linear_model import  Lasso

df = pd.read_csv('forestfires.csv', sep = ',', encoding='latin1')
data = df.values
print(data)


X, y = data[:, :-1], data[:, -1]
print(y)
las = Lasso(alpha= 1.0)

cv = RepeatedKFold(n_splits= 10, n_repeats= 3, random_state= 1)

scores = cross_val_score(las, X, y, scoring= 'neg_mean_absolute_error', cv = cv, n_jobs = -1)
scores = absolute(scores)



