""" Created on Mon May 27 14:04:37 2019 @author: Vasanth """
""" RUL - C-MAPPS Engine data """
#%%
# 1.Basic Header Files
import time
from pandas import read_csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["font.weight"]      = "bold"
plt.rcParams["axes.labelweight"] = "bold"
#%%
#%%
# 2.Loading data

column_names =['Unit','Cycles','OS1','OS2','OS3','S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20','S21']
columns      =['RUL']
data_train   = read_csv('Train_FD001.csv',names=column_names)
data_RUL     = read_csv('RUL_FD001.csv',names=columns)
data_test    = read_csv('Test_FD001.csv' ,names=column_names)
# Drop the sensor values [S1, S5, S6, S10, S16, S18, S19 -constant]
#data_train = data_train.drop(['S1','S5','S6','S10','S16','S18','S19'], axis=1)
#data_test  = data_test.drop (['S1','S5','S6','S10','S16','S18','S19'], axis=1)
#%%
#%%
# 3.Preprocessing data

from sklearn.preprocessing import MinMaxScaler
# ensure all data is float
data_train = data_train.astype('float64')
data_test = data_test.astype('float64')

# normalize features - only the sensor values
scaler = MinMaxScaler(feature_range=(0, 1))
data_train[['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20','S21']] = scaler.fit_transform(data_train[['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20','S21']])
data_test [['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20','S21']] = scaler.fit_transform(data_test [['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20','S21']])

# Convert the array to dataframe so as to add the column names manually
data_train = pd.DataFrame(data_train)
data_test = pd.DataFrame(data_test)

# Add the column names manualy
data_train.columns  = ['Unit','Cycles','OS1','OS2','OS3','S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20','S21']
data_test.columns   = ['Unit','Cycles','OS1','OS2','OS3','S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20','S21']
#%%
#%%
# 4.Grouping and removing the unwanted sensor values
# Make sure the file is comma separated and save it as csv file
# Initilizing the groups to zero
group_test   =[0]*101 
group_train  =[0]*101
cycles_test  =[0]*101
cycles_train =[0]*101
# Grouping them based on the number of engines(which is 100 in this case)
for i in range (1,101):
    group_train[i]  = data_train.loc[(data_train['Unit']==i)]
    group_test[i] = data_test.loc[(data_test['Unit']==i)]    
for i in range (1,101):
    group_test[i].pop('Unit')
    cycles_test[i]=group_test[i].pop('Cycles')
    group_train[i].pop('Unit')
    cycles_train[i]=group_train[i].pop('Cycles')
# Reindexing from 1
for i in range (1,101):
    group_test[i].index = np.arange(1, len(group_test[i]) + 1)
    group_train[i].index = np.arange(1, len(group_train[i]) + 1)
#%%
#%%
# 5. Plot and check the sensor value variations
#for i in range(1,101):
#    f1 = plt.figure(1)
#    plt.plot(group_train[i]['S2'])
#    plt.plot(group_train[i]['S3'])
#    plt.plot(group_train[i]['S4'])
#    plt.plot(group_train[i]['S7'])
#    plt.plot(group_train[i]['S8'])
#    plt.plot(group_train[i]['S9'])
#    plt.plot(group_train[i]['S11'])
#    plt.plot(group_train[i]['S12'])
#    plt.plot(group_train[i]['S13'])
#    plt.plot(group_train[i]['S14'])
#    plt.plot(group_train[i]['S15'])
#    plt.plot(group_train[i]['S17'])
#    plt.plot(group_train[i]['S20'])
#    plt.plot(group_train[i]['S21'])
#    plt.xlabel('Time (Cycles)')
#    plt.ylabel('Sensor Values')
#    plt.title('Sensor Values - Train FD001')
#    plt.legend(['S2', 'S3', 'S4', 'S7','S8','S9','S11','S12','S13','S14','S15','S17','S20','S21'], loc='best')
#    plt.grid(True)
#    f1.show(1)
#%%
#%%    
# 6.Dimensionality reduction    
from sklearn.decomposition import PCA
# a.Very low increase in the values
combo1 = ['S1', 'S2', 'S3', 'S10']
F1   = data_train.loc[:, combo1].values
F1t  = data_test.loc [:, combo1].values
pca  = PCA(n_components=1)
PC  = pca.fit_transform(F1)
PCt = pca.fit_transform(F1t)

## b.Mariginal increase in the values
#combo2 = ['S3','S4']
#F2   = data_train.loc[:, combo2].values
#F2t  = data_test.loc [:, combo2].values
#pca  = PCA(n_components=1)
#PC2  = pca.fit_transform(F2)
#PC2t = pca.fit_transform(F2t)
#
## c.Decrease in the values
#combo3 = ['S7','S12','S20','S21']
#F3   = data_train.loc[:, combo3].values
#F3t  = data_test.loc [:, combo3].values
#pca  = PCA(n_components=1)
#PC3  = pca.fit_transform(F3)
#PC3t = pca.fit_transform(F3t)
#
## d.High increase in the values
#combo4 = ['S9','S14']
#F4   = data_train.loc[:, combo4].values
#F4t  = data_test.loc [:, combo4].values
#pca  = PCA(n_components=1)
#PC4  = pca.fit_transform(F4)
#PC4t = pca.fit_transform(F4t)
#
###converting objects to dataframes and merging it to one value.
#PC1 = pd.DataFrame(PC1)
#PC2 = pd.DataFrame(PC2)
#PC3 = pd.DataFrame(PC3)
#PC4 = pd.DataFrame(PC4)
#
#PC1t = pd.DataFrame(PC1t)
#PC2t = pd.DataFrame(PC2t)
#PC3t = pd.DataFrame(PC3t)
#PC4t = pd.DataFrame(PC4t)
#
#df = pd.concat([PC1,PC2,PC3,PC4],axis=1)
#df.columns = ['PC1','PC2','PC3','PC4']
#finalcombo = ['PC1','PC2','PC3','PC4']
##df = pd.concat([PC1,PC2],axis=1)
##df.columns = ['PC1','PC2']
##finalcombo = ['PC1','PC2']
#Ffinal = df.loc[:,finalcombo].values
#pca = PCA(n_components=1)
#PC  = pca.fit_transform(Ffinal)
#PC  = pd.DataFrame(PC)
#
#dft = pd.concat([PC1t,PC2t,PC3t,PC4t],axis=1)
#dft.columns = ['PC1','PC2','PC3','PC4']
#finalcombo = ['PC1','PC2','PC3','PC4']
##dft = pd.concat([PC1t,PC2t],axis=1)
##dft.columns = ['PC1','PC2']
##finalcombo = ['PC1','PC2']
#Ffinalt = dft.loc[:,finalcombo].values
#pca = PCA(n_components=1)
#PCt  = pca.fit_transform(Ffinalt)
PCt  = pd.DataFrame(PCt)

##converting objects to dataframes and merging it to one value.
#PC1 = pd.DataFrame(PC1)
#PC2 = pd.DataFrame(PC2)
#PC3 = pd.DataFrame(PC3)
#PC4 = pd.DataFrame(PC4)
PC = pd.DataFrame(PC)
#data = pd.concat([PC1,PC2,PC3,PC4], ignore_index=True,axis=1)
## Add Column name
#data.columns = ['PC1','PC2','PC3','PC4']
PC.columns = ['PC']
PCt.columns = ['PC']
#Add the unit and cycle column to the data - merging and droping unwanted columns
#data = pd.concat([data_train,data],axis=1)
#data = data.drop(columns=['OS1','OS2','OS3','S2','S3','S4','S7','S8','S9','S11','S12','S13','S14','S15','S17','S20','S21'])
data_train = pd.concat([data_train,PC],axis=1)
data_train = data_train.drop(columns=['OS1','OS2','OS3','S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20','S21'])
data_test = pd.concat([data_test,PCt],axis=1)
data_test = data_test.drop(columns=['OS1','OS2','OS3','S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20','S21'])
# Again changing the PC values to a minmax_scale
# ensure all data is float

data_train          = data_train.astype('float64')
data_test           = data_test.astype('float64')
scaler              = MinMaxScaler(feature_range=(0, 1))
data_train[['PC']]  = scaler.fit_transform(data_train[['PC']])
data_test [['PC']]  = scaler.fit_transform(data_test[['PC']])
data_train          = pd.DataFrame(data_train)
data_test           = pd.DataFrame(data_test)

# Add the column names manualy
data_train.columns = ['Unit','Cycles','PC']
data_test.columns  = ['Unit','Cycles','PC']

# Filtering nosie using Savitzky–Golay filter
from scipy.signal import savgol_filter
filtered_train = data_train['PC'].values; 
filtered_test  = data_test['PC'].values;
fil_PC_train   = savgol_filter(filtered_train,5,3,mode='constant'); # window size 101, order = 2
fil_PC_train = pd.DataFrame(fil_PC_train)             # convert it to dataframe
fil_PC_train.columns = ['fPC']
fil_PC_test    = savgol_filter(filtered_test,5,3,mode='constant');
fil_PC_test = pd.DataFrame(fil_PC_test)
fil_PC_test.columns = ['fPCt']

data_train = pd.concat([data_train,fil_PC_train], axis=1, ignore_index=True) # merge normal PC with filtered PC
data_train.columns = ['Unit','Cycles','PC','fPC']                            # add column names
data_train = data_train.drop(['PC'],axis=1)                                  # drop normal PC values
data_train.rename(columns={"fPC": "PC"}, inplace=True)                       # rename filtered PC to PC
data_test  = pd.concat([data_test,fil_PC_test],   axis=1, ignore_index=True)
data_test.columns = ['Unit','Cycles','PC','fPCt']   
data_test = data_test.drop(['PC'], axis=1)
data_test.rename(columns={"fPCt": "PC"}, inplace=True)
#%%
#%%
# 7. Plotting and checking the pc values- same as the one used before
# Initilizing the groups to zero
group_data_train  =[0]*101
cycles_data_train =[0]*101
group_data_test   =[0]*101
cycles_data_test  =[0]*101
# Grouping them based on the number of engines(which is 100 in this case)
for i in range (1,101):
    group_data_train[i] =data_train.loc[(data_train['Unit']==i)]
    group_data_test[i]  =data_test.loc[(data_test['Unit']==i)]
for i in range (1,101):
    group_data_train[i].pop('Unit')
    cycles_data_train[i]=group_data_train[i].pop('Cycles')
    group_data_test[i].pop('Unit')
    cycles_data_test[i]=group_data_test[i].pop('Cycles')
# Reindexing from 1
for i in range (1,101):
    group_data_train[i].index = np.arange(1, len(group_data_train[i]) + 1)
    group_data_test[i].index  = np.arange(1, len(group_data_test[i]) + 1)
# Plotting    
for i in range(1,101):
    f2 = plt.figure(2)
    plt.plot(group_data_train[2]['PC'])
    plt.xlabel('Time (Cycles)')
    plt.ylabel('Principle Component')
    plt.title('Principle Components - Train FD001')
    plt.legend(['Principle Component'], loc='best')
    plt.grid(True)
    f2.show(2)
np.savetxt('B.out',group_data_train[2]['PC'])
for i in range(1,101):
    f3 = plt.figure(3)
    plt.plot(group_data_test[15]['PC'])
    plt.xlabel('Time (Cycles)')
    plt.ylabel('Principle Component')
    plt.title('Principle Components - Test FD001')
    plt.legend(['Principle Component'], loc='best')
    plt.grid(True)
    f3.show(3)
np.savetxt('A.out',group_data_test[15]['PC'])    
#%%
#%%
# 8.Exploratory analyses of the maximum number of cycles per unit
#   (Refer: https://www.rrighart.com/blog-gatu/sensor-time-series-of-aircraft-engines)
cyclestrain = data_train.groupby('Unit', as_index=False)['Cycles'].max()
cyclestest  = data_test.groupby ('Unit', as_index=False)['Cycles'].max()  
fig = plt.figure(figsize = (16,12))
fig.add_subplot(1,2,1)
bar_labels = list(cyclestrain['Unit'])
bars = plt.bar(list(cyclestrain['Unit']), cyclestrain['Cycles'], color='red')
plt.ylim([0, 400])
plt.xlabel('Units', fontsize=16)
plt.ylabel('Max. Cycles', fontsize=16)
plt.title('Max. Cycles per unit in trainset', fontsize=16)
plt.xticks(np.arange(min(bar_labels)-1, max(bar_labels)-1, 5.0), fontsize=12)
plt.yticks(fontsize=12)
fig.add_subplot(1,2,2)
bars = plt.bar(list(cyclestest['Unit']), cyclestest['Cycles'], color='grey')
plt.ylim([0, 400])
plt.xlabel('Units', fontsize=16)
plt.ylabel('Max.Cycles', fontsize=16)
plt.title('Max.Cycles per unit in testset', fontsize=16)
plt.xticks(np.arange(min(bar_labels)-1, max(bar_labels)-1, 5.0), fontsize=12)
plt.yticks(fontsize=12)
plt.show() 
#%%    
#%%
# 9.Estimating Remaining useful life : We determine in the trainset for each row 
# the maximum cycles for the particular unit.We use the groupby function to 
# obtain for every unit the maximum, and in turn use pd.merge to bring these 
# values into the original train set:

data_train = pd.merge(data_train, cyclestrain.groupby('Unit', as_index=False)['Cycles'].max(), how='left', on='Unit')
data_train.rename(columns={"Cycles_x": "Cycles", "Cycles_y": "Maxcycles"}, inplace=True)


# Now we determine the time to failure for every row, which is the number of cycles subtracted 
#from the maximum number of cycles in a particular unit.

data_train['Life'] = data_train['Maxcycles'] - data_train['Cycles']

#Time to failure for each unit has a different length, and it would be good to express this in a fraction
#of remaining number of cycles. This starts for a particular unit at 1.00, and goes to 0.00, the point 
#where the engine fails (TTFx). It is in fact similar to scaling, but here it is applied at the unit level. 
#In Python, we can express this in a function:

def fractionLife(d,a):
    return(d.Life[a]-d.Life.min()) / float(d.Life.max()-d.Life.min())

flifex = []
flife  = []

for i in range(int(data_train['Unit'].min()),int(data_train['Unit'].max()+1)):
    d = data_train[data_train.Unit==i]
    d = d.reset_index(drop=True)
    for a in range(len(d)):
        flifex = fractionLife(d, a)
        flife.append(flifex)
ndata_train = data_train.copy()  
ndata_test = data_test.copy()        
ndata_train['Fraction_Life'] = flife
#%%
#%%
# 10. Input and Ouput to the Model
IP_Train  = ndata_train.values[:,0:3]
OP_Train  = ndata_train.values[:  ,5]
IP_Test   = ndata_test.values [:,0:3]

#reshape input to be 3D [samples, timesteps, features]
IP_Train = IP_Train.reshape((IP_Train.shape[0],  1, IP_Train.shape[1]))
IP_Test  = IP_Test.reshape ((IP_Test.shape[0],   1, IP_Test.shape[1]))
print(IP_Train.shape,IP_Test.shape)
#%%
#%%
# 11.Desgining the network
from tensorflow import keras # high-level library for NN,running on top of TF
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error
from math import sqrt
#from keras.regularizers import L1L2
#start_time = time.clock()
#model = Sequential()
#model.add(LSTM(50, dropout=0.1,input_shape=(IP_Train.shape[1], IP_Train.shape[2]),return_sequences=True))
#model.add(Dropout(0.1))
#model.add(LSTM(50,return_sequences=False))
#model.add(Dropout(0.1))
#model.add(Dense(1))
#model.add(Activation('linear'))
#keras.optimizers.RMSprop(lr=0.001, rho=0.8, epsilon=None, decay=0.0)
#model.compile(loss='mse', optimizer='rmsprop')
##The patience parameter is the amount of epochs to check for improvement
#early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
## fit the network
#history = model.fit(IP_Train, OP_Train, epochs=100, batch_size=200,validation_split=0.03,
#                    verbose=2,shuffle=False,callbacks=[early_stop])
#print("--- %s seconds ---" % (time.clock() - start_time))
# Next, we build a deep network. 
# The first layer is an LSTM layer with 100 units followed by another LSTM layer with 50 units. 
# Dropout is also applied after each LSTM layer to control overfitting. 
# Final layer is a Dense output layer with single unit and sigmoid activation since this is a binary classification problem.
# build the network
start_time = time.clock()
MSE=np.zeros(10)
neuron=np.zeros(10)
neuron[0]=np.int(10)
#from keras import regularizers
#from keras.regularizers import L1L2
model = Sequential()
model.add(LSTM(np.int(neuron[0]),input_shape=(IP_Train.shape[1], IP_Train.shape[2]),activation='tanh',return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(np.int(neuron[0]),return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(1, activation='relu'))
keras.optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse', optimizer='adam')
history = model.fit(IP_Train, OP_Train, epochs=30, batch_size=200,validation_split=0.03,
                    verbose=2,shuffle=False)
mse=model.evaluate(IP_Train, OP_Train)
print("--- %s seconds ---" % (time.clock() - start_time))

MSE[0]=mse

best_mse=MSE[0]

best_neuron=neuron[0]

neuron[1]=np.int(15)

model = Sequential()
model.add(LSTM(np.int(neuron[1]),input_shape=(IP_Train.shape[1], IP_Train.shape[2]),activation='tanh',return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(np.int(neuron[1]),return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(1, activation='relu'))
keras.optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse', optimizer='adam')
history = model.fit(IP_Train, OP_Train, epochs=30, batch_size=200,validation_split=0.03,
                    verbose=2,shuffle=False)
mse=model.evaluate(IP_Train, OP_Train)
print("--- %s seconds ---" % (time.clock() - start_time))
MSE[1]=mse

if MSE[1] < best_mse:
            best_mse = MSE[1]
            best_neuron = neuron[1]
import random

c1=0.1

velocity_neurons=0
velocity_change_inertial_nerupns=0            
velocity_change_inertial_neurons = (best_neuron - neuron[1])*c1*random.random()
velocity_neurons =velocity_neurons + velocity_change_inertial_neurons

max_neurons=128

neuron[2] = min(max_neurons,int(neuron[1] + velocity_neurons))
                                                        
# fit the network
for i in range(2,5):
    model = Sequential()
    model.add(LSTM(np.int(neuron[i]),input_shape=(IP_Train.shape[1], IP_Train.shape[2]),activation='tanh',return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(np.int(neuron[i]),return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='relu'))
    keras.optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(IP_Train, OP_Train, epochs=30, batch_size=200,validation_split=0.03,
                        verbose=2,shuffle=False)
    mse=model.evaluate(IP_Train, OP_Train)
    MSE[i]=mse
    if MSE[i] < best_mse:
            best_mse = MSE[i]
            best_neuron = neuron[i]
    velocity_change_inertial_neurons = (best_neuron - neuron[i])*c1*random.random()
    velocity_neurons =velocity_neurons + velocity_change_inertial_neurons



    neuron[i+1] = min(max_neurons,int(neuron[i] + velocity_neurons))
#%%
# 11. a. Predicting the score 
OP_hat = model.predict(IP_Test)
print(OP_hat.min(), OP_hat.max())
ndata_test['Score'] = OP_hat

# same as train dataset
ndata_test = pd.merge(ndata_test, cyclestest.groupby('Unit', as_index=False)['Cycles'].max(), how='left', on='Unit')
ndata_test.rename(columns={"Cycles_x": "Cycles", "Cycles_y": "Maxcycles"}, inplace=True)

# 11. b. RUL
# First we need to estimate the predicted total number of cycles per unit in the test set. 
#This can be done with the following function:

def totalcycles(data):
    return(data['Cycles'] / (1-data['Score']))    
ndata_test['Maxpredcycles'] = totalcycles(ndata_test)

# Subtract the maximum cycles per unit from the predicted total number of cycles in 
# the test set to obtain the RUL, remaining useful lifetime:

def RULfunction(data):
    return(data['Maxpredcycles'] - data['Maxcycles'])
ndata_test['RUL'] = RULfunction(ndata_test)

# Predict RUL in cycles
#The following will compute the RUL per unit (based on the max. cycles) from the
#RUL column that contains predicted values for each row.

t = ndata_test.columns == 'RUL'
ind = [i for i, x in enumerate(t) if x]

predictedRUL = []


for i in range(int(ndata_test['Unit'].min()),int(ndata_test['Unit'].max()+1)):
  
    npredictedRUL=ndata_test[ndata_test.Unit==i].iloc[int(ndata_test[ndata_test.Unit==i].Cycles.max()-1),ind]
    predictedRUL.append(npredictedRUL)

predictedRUL[0:100]
#%%
#%%
plt.figure(figsize = (16, 8))
plt.plot(data_RUL)
plt.plot(predictedRUL)
plt.xlabel('# Unit', fontsize=16)
plt.xticks(fontsize=16)
plt.ylabel('RUL', fontsize=16)
plt.yticks(fontsize=16)
plt.legend(['True RUL','Predicted RUL'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0)
plt.show()

##performance metrics RMSE
mse = 0
mse = mean_squared_error(data_RUL,predictedRUL)
rmse = sqrt(mse)
print(rmse)

##preformance score 
predictedRUL= pd.DataFrame(predictedRUL) 
#Reindexing from 1
data_RUL.index = np.arange(1, len(data_RUL)+1)
predictedRUL.index = np.arange(1, len(predictedRUL) + 1)
# for caluclating di
diff_RUL = predictedRUL['RUL'] - data_RUL['RUL']
s =[]
r =[]
t =[]
a1 = 10
a2 = 13
s  = 0
for i in range (1,100):
    if diff_RUL[i] < 0:
        s =  [np.exp(-diff_RUL[i]/a1)-1]
        r = s + r
        i = i + 1
    else:
        s = [np.exp(diff_RUL[i]/a2)-1]
        t = s + t
        i = i + 1
score = sum (r + t)
print(score)

#%%
#%%
#Inspecting the model
#model.output_shape
model.summary()
#model.get_config()=
#model.get_weights()
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#%%