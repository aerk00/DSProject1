import pandas as pd
import gplearn
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('forestfires.csv', sep = ',', encoding='latin1')

X, y = train_test_split(df, test_size=.2, random_state= 2)

Y_train = X.iloc[:,-1].values


#https://www.mygreatlearning.com/blog/understanding-of-lasso-regression/