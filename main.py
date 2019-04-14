#Afaf Taik 
#GEI723
#Main function 

#from keras import backend as K
#from keras.engine.topology import Layer
from keras.layers import RNN, Activation,LSTM
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
import keras

import numpy
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from vanilla import VLSTM
from nig import NIGLSTM
from nog import NOGLSTM
from nfg import NFGLSTM

##########################################################################################
#Parameters
n_lag = 5
n_seq = 3 
n_epochs = 40
n_neurones = 40
filename='house_by_minute_1_sum.csv'


###########################################################################################
#Load Data
#Code adapted from machinelearningmastery tutorials
###########################################################################################
# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# load the dataset
dataframe = pd.read_csv(filename, usecols=[0,1], engine='python',infer_datetime_format=True, parse_dates=['DateTime'], index_col=['DateTime'])
hourly_groups = dataframe.resample('H')
hourly_data = hourly_groups.sum()
dataset = hourly_data.values
dataset = hourly_data.astype('float32')
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
traind, testd = dataset.iloc[0:train_size,:], dataset.iloc[train_size:len(dataset),:]

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
traind = scaler.fit_transform(traind)
testd = scaler.transform(testd)

#Turn the data into the supervised learning shape  

train,test = series_to_supervised(traind, n_lag, n_seq), series_to_supervised(testd, n_lag, n_seq)
train_values = train.values
test_values = test.values

# split into train and test sets
trainX,trainY= train_values[:, 0:n_lag], train_values[:, n_lag:]
testX,testY= test_values[:, 0:n_lag], test_values[:, n_lag:]
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1],1))
testX = numpy.reshape(testX, (testX.shape[0],testX.shape[1],1))


###########################################################################################
#Create the models, and fit them
###########################################################################################

# create and fit the LSTM networks
modelV = Sequential()
modelI = Sequential()
modelF = Sequential()
modelO = Sequential()
steps = n_lag
inp = keras.Input((steps, 1))
dense_def = Dense(n_seq)
activation_def = Activation('tanh')


#Vanilla
cellV=VLSTM(n_neurones)
lstm_defV=RNN(cellV)
lstmV = lstm_defV(inp)
denseV = dense_def(lstmV)
activateV = activation_def(denseV)
outV = activateV
modelV = Model([inp], [outV])
modelV.compile(loss='mean_squared_error', optimizer='adam')
print('------------------------Training Vanilla----------------------------')
historyV=modelV.fit(trainX, trainY, validation_data=(testX,testY), epochs=n_epochs, batch_size=16, verbose=2)
print('Done training Vanilla')

#NOG
cellO=NOGLSTM(n_neurones)
lstm_defO=RNN(cellO)
lstmO = lstm_defO(inp)
denseO = dense_def(lstmO)
activateO = activation_def(denseO)
outO = activateO
modelO = Model([inp], [outO])
modelO.compile(loss='mean_squared_error', optimizer='adam')
print('------------------------Training No Output Gate LSTM----------------------------')
historyO=modelO.fit(trainX, trainY, validation_data=(testX,testY), epochs=n_epochs, batch_size=16, verbose=2)
print('Done training NOG')

#NFG
cellF=NFGLSTM(n_neurones)
lstm_defF=RNN(cellF)
lstmF = lstm_defF(inp)
denseF = dense_def(lstmF)
activateF = activation_def(denseF)
outF = activateF
modelF = Model([inp], [outF])
modelF.compile(loss='mean_squared_error', optimizer='adam')
print('------------------------Training No Forget Gate LSTM----------------------------')
historyF=modelF.fit(trainX, trainY, validation_data=(testX,testY), epochs=n_epochs, batch_size=16, verbose=2)
print('Done training NFG')

#NIG
cellI=NIGLSTM(n_neurones)
lstm_defI=RNN(cellI)
lstmI = lstm_defI(inp)
denseI = dense_def(lstmI)
activateI = activation_def(denseI)
outI = activateI
modelI = Model([inp], [outI])
modelI.compile(loss='mean_squared_error', optimizer='adam')
print('------------------------Training No Input Gate LSTM----------------------------')
historyI=modelI.fit(trainX, trainY, validation_data=(testX,testY), epochs=n_epochs, batch_size=16, verbose=2)
print('Done training NIG')

#Plot results
plt.figure(1)
plt.plot(historyV.history['val_loss'])
plt.plot(historyO.history['val_loss'])
plt.plot(historyI.history['val_loss'])
plt.plot(historyF.history['val_loss'])
plt.ylabel('loss for test sets')
plt.xlabel('epoch')
plt.legend(['Vanilla','NOG','NIG','NFG'])
plt.show()

plt.figure(2)
plt.plot(historyV.history['loss'])
plt.plot(historyO.history['loss'])
plt.plot(historyI.history['loss'])
plt.plot(historyF.history['loss'])
plt.ylabel('loss for train sets')
plt.xlabel('epoch')
plt.legend(['Vanilla','NOG','NIG','NFG'])
plt.show()
