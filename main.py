import pandas as pd
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('forestfires.csv', sep = ',', encoding='latin1')

data_train, data_val = train_test_split(df, test_size = 0.2, random_state = 2)

#https://www.mygreatlearning.com/blog/understanding-of-lasso-regression/
#https://gplearn.readthedocs.io/en/stable/