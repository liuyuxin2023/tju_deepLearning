from pandas_datareader import data as web
import numpy as np
import pandas as pd
from sklearn import mixture as mix
import seaborn as sns 
import matplotlib.pyplot as plt
import talib as ta
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import fix_yahoo_finance

dataset= pd.read_csv('AL8888.XSGE.csv')
dataset = dataset.dropna(how ='any')
dataset2= pd.read_csv('AL8888.XSGE.csv')
dataset2 = dataset2.dropna(how ='any')
#dataset2['Price_Rise'] = np.where(dataset2['Close'].shift(-1) > dataset2['Close'], 1, 0)
new_data02=dataset2.iloc[:, 1: 6]
new_data02.dropna()
split=int(0.75*len(new_data02))
new_data02.iloc[:split].to_csv('train2.csv')
new_data02.iloc[split:].to_csv('test2.csv')
print(new_data02.head())

