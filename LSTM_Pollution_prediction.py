""" Created on Mon Mar 25 18:43:45 2019 @author: Vasanth """
""" Pollution_PM_2.5 """
#%%
import time
from pandas import read_csv
import pandas as pd
import numpy as np
from matplotlib import pyplot
pyplot.rcParams["axes.titleweight"] = "bold"
pyplot.rcParams["font.weight"]      = "bold"
pyplot.rcParams["axes.labelweight"] = "bold"
#%%
#%%
#load data with indexing(index_col = 0 is not used)
data = read_csv('Pollution_Input_2014_17_Mis.csv')
# Dropping the unwanted rows (axis =0)
#data = data.drop([0,1,2,3,4,5,6,7,8,9,10,11,12,13], axis=0)
# Dropping the unwanted rows using ilocator
data = data.iloc[14:,]
# Dropping the unwanted column 
data = data.iloc[:,2:]
# Dropping the unwanted last n rows
#data.drop(data.tail(7).index,inplace=True)
# Manually entering columns names
data.columns = ['Date','H1', 'H2', 'H3','H4','H5','H6','H7','H8','H9','H10',
                'H11','H12','H13','H14','H15','H16','H17','H18','H19','H20',
                'H21','H22','H23','H24']
# Reindexing from 1
data.index = np.arange(1, len(data) + 1)
# Droping the date column
data = data.drop(columns='Date')
# Since the datatype is object it is converted to float64
data = data.apply(pd.to_numeric)
# The data has missing and invalid data - replaced by NaN
data = data.replace([-999,9999], np.NAN)
# Replaced nan values are filled with mean 
data.fillna(data.mean(), inplace=True)
#data['dates'] = pd.Timestamp('2016-11-06')
# This converts 24hrs data in single series as required
stacked_data = data.stack()
# Dropping the unwanted column 
#stacked_data = stacked_data.iloc[:,0:]
# Adding date and time with hourly frequency 
timeframe = pd.date_range(start="1-Jan-2014", end="1-Jan-2018", freq='H') 
leap = []
for each in timeframe:
    if each.month==2 and each.day ==29:
        leap.append(each)
timeframe = timeframe.drop(leap)
# datetimeindex to dataframe with changing the index
timeframe = timeframe.to_frame(index=False)
# Reindexing from 1
timeframe.index = np.arange(1, len(timeframe) + 1)
timeframe.columns = ['Date']
#Dropping the unwanted last n rows
timeframe.drop(timeframe.tail(1).index,inplace=True)
stacked_data.to_csv('Pollution_Output_2014_17_Mis.csv', header = 'false')
stacked_data = read_csv('Pollution_Output_2014_17_Mis.csv')
# Reindexing from 1
stacked_data.index = np.arange(1, len(stacked_data) + 1)
# Naming the columns in the file
stacked_data.columns = ['Day number','Hour','PM2.5']
# Dropping the columns day number and hour
stacked_data = stacked_data.drop(columns=["Day number","Hour"])
# Merging it with date and time
dataset = pd.concat([timeframe,stacked_data],axis=1)
# Naming the index in the file
dataset.index.name = 'Index'
dataset.to_csv('Pollution_Output_2014_17_Mis.csv')
#%%
#%%
f1 = dataset.plot(x='Date', y='PM2.5',color='blue', linestyle='-',linewidth=1)
pyplot.xlabel('Month (2014-2017)')
pyplot.ylabel('Fine particulate matter (PM2.5)')
pyplot.title('DataSet - Mississauga')
pyplot.margins(x=0,y=0)
pyplot.ylim(0,70)
pyplot.yticks(np.arange(0,70,10))
pyplot.grid()
pyplot.savefig('f1.svg',dpi=2540)
pyplot.savefig('f1.png',dpi=2540)
#pyplot.clf()
#%%
#%%
from sklearn.preprocessing import MinMaxScaler
# ensure all data is float
PM_values = dataset.drop(columns='Date')
# normalize features
scaler = MinMaxScaler(feature_range=(0,1))
Norm_PM_values = scaler.fit_transform(PM_values)
#%%
#%%
from pandas import DataFrame
from pandas import concat
# frame as supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)'    % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
Supervised_Norm_PM = series_to_supervised(Norm_PM_values,1,1)
f = open("Pollution_ASupervised.txt","w+")
Supervised_Norm_PM.to_csv("Pollution_ASupervised.txt", sep='\t', encoding='utf-8')
#print(Supervised_Norm_PM.head())
#%%
#%%
#Split the data for training(2014,2015), validation(2016), testing(2017)
PM_values     = Supervised_Norm_PM.values
Nhrs_train    = 2*365*24
Train         = PM_values[:Nhrs_train, :]
Nhrs_vali     = 365*24
Ttl_Nhrs_vali = 3*365*24
Vali          = PM_values[Nhrs_train:Ttl_Nhrs_vali, :]
Test          = PM_values[Ttl_Nhrs_vali:,:]
#split into input and outputs
Train_IP,Train_OP = Train[:,:-1],Train[:,-1]
Vali_IP ,Vali_OP  = Vali[:,:-1] ,Vali[:,-1]
Test_IP ,Test_OP  = Test[:,:-1] ,Test[:,-1]
#reshape input to be 3D [samples, timesteps, features]
Train_IP = Train_IP.reshape((Train_IP.shape[0],  1, Train_IP.shape[1]))
Vali_IP  = Vali_IP.reshape ((Vali_IP.shape[0],   1, Vali_IP.shape[1]))
Test_IP  = Test_IP.reshape ((Test_IP.shape[0],   1, Test_IP.shape[1]))
print(Train_IP.shape, Train_OP.shape, Vali_IP.shape, Vali_OP.shape, Test_IP.shape, Test_OP.shape)
#%%
#%%
#Desgining the network
from tensorflow import keras # high-level library for NN,running on top of TF
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import mean_absolute_error
from math import sqrt
model = Sequential()
model.add(LSTM(50, dropout=0.2,input_shape=(Train_IP.shape[1], Train_IP.shape[2])))
model.add(Dense(1))
model.compile(loss = 'mae', optimizer = 'adam')
model.compile(loss = 'mse', optimizer = 'adam')
#The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
#fit network
start_time = time.clock()
history = model.fit(Train_IP, Train_OP, epochs=100, batch_size=64, validation_data=(Vali_IP, Vali_OP),
                    verbose=1,shuffle=False,callbacks=[early_stop])
print("--- %s seconds ---" % (time.clock() - start_time))
#plot history for training and validation loss values
#%%
#%%
f2 = pyplot.figure(2)
#pyplot.plot(history.history['loss'],color='black',linewidth=1,linestyle='-.')
#pyplot.plot(history.history['val_loss'],color='black',linewidth=1,linestyle='-')
pyplot.plot(history.history['loss'],linewidth=1,linestyle='-.',color = 'k')
pyplot.plot(history.history['val_loss'],linewidth=1,linestyle='-', color = 'k')
pyplot.xlabel('Epoch')
pyplot.ylabel('Mean Absolute Error')
pyplot.title('Model Loss - Missusauga')
pyplot.legend(['Train', 'Validation'], loc='best')
pyplot.margins(x=0)
#pyplot.xlim(0,10)
#pyplot.xticks(np.arange(0,10,1))
pyplot.grid(b=True, which='major', color='#666666', linestyle='-')
# Show the minor grid lines with very faint and almost transparent grey lines
pyplot.minorticks_on()
pyplot.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
pyplot.savefig('f2.svg',dpi=2540)
pyplot.savefig('f2.png',dpi=2540)
f2.show(2)
#pyplot.clf()
#%%
#%%
f3 = pyplot.figure(3)
#pyplot.plot(history.history['loss'],color='black',linewidth=1,linestyle='-.')
#pyplot.plot(history.history['val_loss'],color='black',linewidth=1,linestyle='-')
pyplot.plot(history.history['loss'],linewidth=1,linestyle='-.', color = 'k')
pyplot.plot(history.history['val_loss'],linewidth=1,linestyle='-', color = 'k')
pyplot.xlabel('Epoch')
pyplot.ylabel('Mean Squared Error')
pyplot.title('Model Loss - Missusauga')
pyplot.legend(['Validation', 'Train'], loc='best')
pyplot.margins(x=0)
#pyplot.xlim(0,10)
#pyplot.xticks(np.arange(0,10,1))
pyplot.grid(b=True, which='major', color='#666666', linestyle='-')
# Show the minor grid lines with very faint and almost transparent grey lines
pyplot.minorticks_on()
pyplot.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
pyplot.savefig('f3.svg',dpi=2540)
pyplot.savefig('f3.png',dpi=2540)
f3.show(3)
#%%
#%%
#Evaluate the model
yhat = model.predict(Test_IP)
Test_IP = Test_IP.reshape((Test_IP.shape[0], Test_IP.shape[2]))
# invert scaling for forecast
inv_yhat = scaler.inverse_transform(yhat)
# invert scaling for actual
Test_OP = Test_OP.reshape((len(Test_OP), 1))
inv_y = scaler.inverse_transform(Test_OP)
#%%
#%%
f4 = pyplot.figure(4)
#pyplot.plot(v1)
#pyplot.plot(v2)
pyplot.plot(inv_y, color = 'k', linestyle = '-.')
pyplot.plot(inv_yhat,color = 'k', linestyle = '--')
pyplot.xlabel('Time (Hrs) - 2017')
pyplot.ylabel('Fine particulate matter (PM2.5)')
pyplot.title('Comparison Plot - Mississauga')
pyplot.legend(['Actual PM2.5', 'Predicted PM2.5'], loc='best')
#np.savetxt('A.out',inv_y)
#np.savetxt('B.out',inv_yhat)
pyplot.xlim(0,8760)
pyplot.ylim(0,)
#pyplot.xticks(np.arange(0,17500,2500))
pyplot.grid()
sub_axes = pyplot.axes([.48, .6, .25, .25]) 
sub_axes.plot(inv_y[1:100])
sub_axes.plot(inv_yhat[1:100])
pyplot.savefig('f4.svg',dpi=2540)
pyplot.savefig('f4.png',dpi=2540)
f4.show(4)
np.savetxt('A.out',inv_y)
np.savetxt('B.out',inv_yhat)
#%%
#%%
f5 = pyplot.figure(5)
x = inv_y
y = inv_yhat
pyplot.scatter(x,y, color = 'k')
pyplot.xlabel('Actual PM(2.5)')
pyplot.ylabel('Predicted PM(2.5)')
pyplot.title('Actual Vs Predicted - Mississauga')
pyplot.xlim(0,40)
pyplot.ylim(0,40)
_ = pyplot.plot([0, 40], [0, 40],color = 'k')
pyplot.grid()
pyplot.savefig('f5.svg',dpi=2540)
pyplot.savefig('f5.png',dpi=2540)
f5.show(5)
#%%
#%%
# Performance metrics
#calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
#forecast_error = [inv_y[i]-inv_yhat[i] for i in range(len(inv_y))]
#mean_sqared_error = np.mean((forecast_error^2), dtype=np.float64)
#rmse = sqrt(mean_sqared_error)
#rms = sqrt(mean_squared_error(inv_y, inv_yhat))
#print('Test RMSE: %.3f' % rms)
#calculate Percentage Error
forecast_errors = [inv_y[i]-inv_yhat[i] for i in range(len(inv_y))]
n_mean_abs_error = np.mean((np.abs(forecast_errors)), dtype=np.float64)
d_mean_abs_error = np.mean(inv_y, dtype=np.float64)
m_a_e =(n_mean_abs_error/d_mean_abs_error)*100
print('Forecast Errors: %s' % m_a_e)
#%%
#%%
#Inspecting the model
#model.output_shape
model.summary()
#model.get_config()
#model.get_weights()
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#%%
