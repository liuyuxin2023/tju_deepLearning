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
import math
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
price_seri=inputs[:, 1]
price_seri1=price_seri.reshape(price_seri.shape[0],)
price_seri0=price_seri1[60:]
print('real price seri shape: ',len(price_seri0) )

'''
部分代码脱敏，请谅解
'''
def cal_ret(ser):
    return pd.Series(ser).pct_change()
# calculate the win_rate of the predicted return
def win_rate(pred, real):
    res = round(sum(np.sign(pred)==np.sign(real))/len(pred), 3)
    print(res)
    return res
    
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
j=17
real0d=df['realPrice_sc'].shift(1).rolling(window = j).mean()
real20std=df['realPrice_sc'].shift(1).rolling(window = 20).std()
real20mean=df['realPrice_sc'].shift(1).rolling(window = 20).mean()
#print(real20std[:50])
#print(real20mean[:50])
real20std=np.array(real20std)
real20mean=np.array(real20mean)
threshold=[]
for i in range(len(real20std)):
	x=real20std[i]*1.0 / real20mean[i]
	if not math.isnan(x):
		threshold.append(x)
	else:
		threshold.append(0.015)

print('thredshod, ',threshold[0])
print(threshold[:50])

pred0d=df['predPrice_sc'].shift(1).rolling(window = j).mean()
real0d=np.array(real0d)
pred0d=np.array(pred0d)
	
pred_ret = cal_ret(pred0d)
rel_ret = cal_ret(real0d)
pred_ret = pred_ret[~np.isnan(pred_ret)]
rel_ret = rel_ret[~np.isnan(rel_ret)]
win = win_rate(pred_ret, rel_ret)
np.corrcoef(pred_ret, rel_ret)
	
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
print('bin_length: ',len(real_bin))
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
	
plt.plot(real0, color = 'red', label = 'Real Future Price')
plt.plot(pred0, color = 'blue', label = 'Predicted FuturePrice')
plt.title('AL8888.XSGE price Prediction')
plt.xlabel('Time')
plt.ylabel('AL8888 Price')
plt.legend()
plt.savefig('LSTM_compare_avg17.png')

holding=0
mrk_value=0
net_cash_flow=0
profit_list=[]
buy_flag=False
base_line=pred0d[0]
un_buy=0
def buy(cur_price):
	global holding
	global mrk_value
	global net_cash_flow
	global buy_flag
	global base_line
	global un_buy
	holding=1000
	mrk_value=holding*cur_price*100
	net_cash_flow=net_cash_flow-holding*cur_price
	buy_flag=True
	un_buy=0
	print('Start to buy this future in shares of',holding*100)

def sell_out(cur_price):
	global holding
	global mrk_value
	global net_cash_flow
	global buy_flag
	global base_line
	global un_buy
	if holding<500:
		net_cash_flow+=(cur_price*holding*100)
		mrk_value=0
		holding=0
		buy_flag=False
		un_buy=0
		print('sell out all holding: ')
	else: 
		net_cash_flow+=(500*100*cur_price)
		holding-=500
		mrk_value=holding*cur_price*100
		un_buy=0
		print('sell out some holding: ',500,' more shares of future')
		
def add_holding(cur_price):
	global holding
	global mrk_value
	global net_cash_flow
	global buy_flag
	global base_line
	global un_buy
	add_holding=500
	net_cash_flow-=(add_holding*100*cur_price)
	holding+=add_holding
	mrk_value=holding*cur_price*100
	un_buy=0
	print('Add more holding: ',add_holding*100,' more shares of future')

for i in range(len(pred_bin)):
	un_buy=un_buy+1
	if pred0d[i+1]/base_line > (1+threshold[i]):
		base_line=pred0d[i]
		if buy_flag:
			add_holding(price_seri0[i])
		else: 
			buy(price_seri0[i])
			
	elif pred0d[i+1]/base_line < (1-threshold[i]):
		base_line=pred0d[i]
		if buy_flag:
			sell_out(price_seri0[i])
			
	else: 
		if un_buy>=30:
			base_line=pred0d[i]
			un_buy=0
	profit_list.append(net_cash_flow+mrk_value)
	
df2=pd.DataFrame(profit_list)
df2.to_csv('profit_record.csv')
plt.figure()
plt.plot(profit_list, color = 'blue', label = 'profit')
plt.title('profit of the strategy')
plt.xlabel('Time')
plt.ylabel('profit')

plt.savefig('profit_graph.png')	
plt.show()




