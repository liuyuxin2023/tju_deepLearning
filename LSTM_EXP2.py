import numpy as np
import pandas as pd
from sklearn import metrics
import sklearn
#import seaborn as sns 
import matplotlib.pyplot as plt
import talib as ta
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model
import os.path
dataset_train = pd.read_csv('train2.csv')
training_set = dataset_train.iloc[:, 1:].values
#print(training_set.head() )

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, :])
    y_train.append(training_set_scaled[i, 1])
X_train, y_train = np.array(X_train), np.array(y_train)
#print(X_train.shape)
#print(y_train[:5])
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]*5, 1))


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
print(X_train.shape)
print(y_train.shape)

if not (os.path.isfile('LSTM_model_exp.h5')):
	regressor.fit(X_train, y_train, epochs = 2, batch_size = 32)
	regressor.save('LSTM_model_exp.h5')
else: 
	del regressor
	regressor= load_model('LSTM_model_exp.h5')

dataset_test = pd.read_csv('test2.csv')
dataset_total = pd.concat((dataset_train.iloc[:,1:], dataset_test.iloc[:,1:]), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
#print(inputs.shape)
inputs = inputs.reshape(-1,5)
#print(inputs.shape)

inputs = sc.transform(inputs)
#print(inputs.shape)
real_stock_price = inputs[:, 1]
X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, :])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]*5, 1))

'''
部分代码脱敏，请谅解
'''
    
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
for j in range(5,20):
	real0d=#脱敏，请谅解#mean()
	pred0d=#脱敏，请谅解#mean()
	real0d=np.array(real0d)
	pred0d=np.array(pred0d)
	#real0=real1[:len(real1)]
	#pred0=pred1[3:]


    
	pred_ret = cal_ret(pred0d)
	rel_ret = cal_ret(real0d)
	pred_ret = pred_ret[~np.isnan(pred_ret)]
	rel_ret = rel_ret[~np.isnan(rel_ret)]
	win = win_rate(pred_ret, rel_ret)
	np.corrcoef(pred_ret, rel_ret)
	
	#print(predicted_stock_price.shape)

	#predicted_stock_price = sc.inverse_transform(predicted_stock_price)
	real_bin=[]
	pred_bin=[]
	for i in range(len(real0d)-1):
		if real0d[i]<real0d[i+1]:
			real_bin.append(1)
		else:
			real_bin.append(0)
		if pred0d[i]<pred0d[i+1]:
			pred_bin.append(1)
		else:
			pred_bin.append(0)
	accuracy = accuracy_score(real_bin, pred_bin)
	print('accuracy ', accuracy)

	precision = precision_score(real_bin, pred_bin, average='macro') 
	print('precision: ',precision)

	recall=recall_score(real_bin, pred_bin, average='macro')
	print('recall: ',recall)

	f1=f1_score(real_bin, pred_bin, average='macro')
	print('f1: ',f1)
	ROC_AUC=roc_auc_score(real_bin, pred_bin, average='macro')
	print('ROC/AUCa: ',ROC_AUC)
	if accuracy > max_score:
		max_score=accuracy
		scores=[accuracy,precision,recall,f1,ROC_AUC]
		opt_real_avg=real0d
		opt_pred_avg=pred0d
		max_window=j

print('the window with the optional accuracy: ', max_window)
print('max_accuracy ', scores[0])
print('_precision: ',scores[1])
print('_recall: ',scores[2])
print('_f1: ',scores[3])
print('_ROC/AUCa: ',scores[4])











