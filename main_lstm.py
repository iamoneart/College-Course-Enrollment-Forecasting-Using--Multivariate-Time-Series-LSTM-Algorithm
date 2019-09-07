from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from matplotlib.ticker import FormatStrFormatter
from pandas import read_csv
from pandas import DataFrame
from pandas import concat 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import CSVLogger
from keras import backend as K
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick


# This convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# defines the input sequence (t-n, ... t-1)
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
	# combining all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# rows with NaN values are dropped
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# load dataset
dataset = read_csv('C:\\Users\\foyeleke\\Desktop\\Test\\predcourseproj2\\crse_proj_data.csv', header=0, index_col=0)
values = dataset.values

# make sure all data is of type float
values = values.astype('float32')

# features are normalized
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)

# leave out columns that we don't want to predict
reframed.drop(reframed.columns[[3]], axis=1, inplace=True)
print(reframed.head(n=12))

# split the data into train and test sets
values = reframed.values
n_train_months = int(len(values)* 0.5) # <== 60 min * 24 hrs * 182 days (6 months) / 15 minutes level interval
train = values[:n_train_months, :]
test = values[n_train_months:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network architecture
model = Sequential()
model.add(LSTM(2, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
csv_logger = CSVLogger('C:\\Users\\foyeleke\\Desktop\\Test\\predcourseproj2\\sept1 _log.csv', append=True, separator=',')
history = model.fit(train_X, train_y, callbacks=[csv_logger], epochs=200, batch_size= 8, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# show history with plot
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)#python main_lstm_program.py
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
print(inv_yhat)

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
print(inv_y)

# compute the Root-Mean-Square-Error
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

np.savetxt('C:\\Users\\foyeleke\\Desktop\\Test\\predcourseproj2\\sept1_result.csv', inv_yhat, delimiter=',')
#temp_data =read_csv('C:\\Users\\foyeleke\\Desktop\\Multivariate_Result\\1_parameter\\1_Param_results\\out171s.csv', header=0, usecols=[2])





