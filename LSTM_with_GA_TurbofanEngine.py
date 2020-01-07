""" Created on Wed Jun 5 12:06:32 2019 and updated Jun 6 @author: Vasanth """
""" RUL - C-MAPPS Engine data """
#%%
# 1.Basic Header Files
#import time
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
data_RUL     = read_csv('RUL_FD001.csv',  names=columns)
data_test    = read_csv('Test_FD001.csv' ,names=column_names)
# Drop the sensor values [S1, S5, S6, S10, S16, S18, S19 -constant]
data_train = data_train.drop(['S1','S5','S6','S10','S16','S18','S19'], axis=1)
data_test  = data_test.drop (['S1','S5','S6','S10','S16','S18','S19'], axis=1)
#%%
#%%
# 3.Preprocessing data

from sklearn.preprocessing import MinMaxScaler
# ensure all data is float
data_train = data_train.astype('float64')
data_test = data_test.astype('float64')

# normalize features - only the sensor values
scaler = MinMaxScaler(feature_range=(0, 1))
data_train[['S2','S3','S4','S7','S8','S9','S11','S12','S13','S14','S15','S17','S20','S21']] = scaler.fit_transform(data_train[['S2','S3','S4','S7','S8','S9','S11','S12','S13','S14','S15','S17','S20','S21']])
data_test [['S2','S3','S4','S7','S8','S9','S11','S12','S13','S14','S15','S17','S20','S21']] = scaler.fit_transform(data_test [['S2','S3','S4','S7','S8','S9','S11','S12','S13','S14','S15','S17','S20','S21']])

# Convert the array to dataframe so as to add the column names manually
data_train = pd.DataFrame(data_train)
data_test  = pd.DataFrame(data_test)

# Add the column names manualy
data_train.columns  = ['Unit','Cycles','OS1','OS2','OS3','S2','S3','S4','S7','S8','S9','S11','S12','S13','S14','S15','S17','S20','S21']
data_test.columns   = ['Unit','Cycles','OS1','OS2','OS3','S2','S3','S4','S7','S8','S9','S11','S12','S13','S14','S15','S17','S20','S21']
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
# 5.Dimensionality reduction    
from sklearn.decomposition import PCA
# a.Very low increase in the values
combo1 = ['S2', 'S8', 'S11', 'S13', 'S15', 'S17']
F1   = data_train.loc[:, combo1].values
F1t  = data_test.loc [:, combo1].values
pca  = PCA(n_components=1)
PC1  = pca.fit_transform(F1)
PC1t = pca.fit_transform(F1t)

# b.Mariginal increase in the values
combo2 = ['S3','S4']
F2   = data_train.loc[:, combo2].values
F2t  = data_test.loc [:, combo2].values
pca  = PCA(n_components=1)
PC2  = pca.fit_transform(F2)
PC2t = pca.fit_transform(F2t)

# c.Decrease in the values
combo3 = ['S7','S12','S20','S21']
F3   = data_train.loc[:, combo3].values
F3t  = data_test.loc [:, combo3].values
pca  = PCA(n_components=1)
PC3  = pca.fit_transform(F3)
PC3t = pca.fit_transform(F3t)

# d.High increase in the values
combo4 = ['S9','S14']
F4   = data_train.loc[:, combo4].values
F4t  = data_test.loc [:, combo4].values
pca  = PCA(n_components=1)
PC4  = pca.fit_transform(F4)
PC4t = pca.fit_transform(F4t)

##converting objects to dataframes and merging it to one value.
PC1 = pd.DataFrame(PC1)
PC2 = pd.DataFrame(PC2)
PC3 = pd.DataFrame(PC3)
PC4 = pd.DataFrame(PC4)

PC1t = pd.DataFrame(PC1t)
PC2t = pd.DataFrame(PC2t)
PC3t = pd.DataFrame(PC3t)
PC4t = pd.DataFrame(PC4t)

df = pd.concat([PC1,PC2,PC3,PC4],axis=1)
df.columns = ['PC1','PC2','PC3','PC4']
finalcombo = ['PC1','PC2','PC3','PC4']
Ffinal = df.loc[:,finalcombo].values
pca = PCA(n_components=1)
PC  = pca.fit_transform(Ffinal)
PC  = pd.DataFrame(PC)

dft = pd.concat([PC1t,PC2t,PC3t,PC4t],axis=1)
dft.columns = ['PC1','PC2','PC3','PC4']
finalcombo = ['PC1','PC2','PC3','PC4']  
Ffinalt = dft.loc[:,finalcombo].values
pca = PCA(n_components=1)
PCt  = pca.fit_transform(Ffinalt)
PCt  = pd.DataFrame(PCt)

PC.columns  = ['PC']
PCt.columns = ['PC']  

data_train = pd.concat([data_train,PC],axis=1)
data_train = data_train.drop(columns=['OS1','OS2','OS3','S2','S3','S4','S7','S8','S9','S11','S12','S13','S14','S15','S17','S20','S21'])
data_test  = pd.concat([data_test,PCt],axis=1)
data_test  = data_test.drop(columns=['OS1','OS2','OS3','S2','S3','S4','S7','S8','S9','S11','S12','S13','S14','S15','S17','S20','S21'])
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
fil_PC_train   = pd.DataFrame(fil_PC_train)             # convert it to dataframe
fil_PC_train.columns = ['fPC']
fil_PC_test          = savgol_filter(filtered_test,5,3,mode='constant');
fil_PC_test        = pd.DataFrame(fil_PC_test)
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
# 6.Plotting and checking the pc values- same as the one used before
# Initilizing the groups to zero
group_data_train  =[0]*101
cycles_data_train =[0]*101
group_data_test   =[0]*101
cycles_data_test  =[0]*101
# Grouping them based on the number of engines(which is 100 in this case)
for i in range (1,101):
    group_data_train[i] = data_train.loc[(data_train['Unit']==i)]
    group_data_test[i]  = data_test.loc[(data_test['Unit']==i)]
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
    f1 = plt.figure(1)
    plt.plot(group_data_train[i]['PC'])
    plt.xlabel('Time (Cycles)')
    plt.ylabel('Principle Component')
    plt.title('Principle Components - Train FD001')
    plt.legend(['Principle Component'], loc='best')
    plt.grid(True)
    f1.show(1)
for i in range(1,101):
    f2 = plt.figure(2)
    plt.plot(group_data_test[i]['PC'])
    plt.xlabel('Time (Cycles)')
    plt.ylabel('Principle Component')
    plt.title('Principle Components - Test FD001')
    plt.legend(['Principle Component'], loc='best')
    plt.grid(True)
    f2.show(2)
#%% 
#%%
# 7.Estimating Remaining useful life : We determine in the trainset for each row 
# the maximum cycles for the particular unit.We use the groupby function to 
# obtain for every unit the maximum, and in turn use pd.merge to bring these 
# values into the original train set:
cyclestrain = data_train.groupby('Unit', as_index=False)['Cycles'].max()
cyclestest  = data_test.groupby ('Unit', as_index=False)['Cycles'].max()  
data_train  = pd.merge(data_train, cyclestrain.groupby('Unit', as_index=False)['Cycles'].max(), how='left', on='Unit')
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
IP_Train  = ndata_train.values [0:10000    ,0:3]
OP_Train  = ndata_train.values [0:10000    ,5]
IP_Test   = ndata_train.values [10001:13096,0:3] #23% percentage
OP_Test   = ndata_train.values [10001:13096,5]

#reshape input to be 3D [samples, timesteps, features]
IP_Train = IP_Train.reshape((IP_Train.shape[0],  1, IP_Train.shape[1]))
IP_Test  = IP_Test.reshape ((IP_Test.shape[0],   1, IP_Test.shape[1]))
print(IP_Train.shape,IP_Test.shape)
#%%  
#%%
# 11.Desgining the network
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation
#from keras.layers import Dropout
import random

# Parameters for the GA algorithm
classes     = 1
batch_size  = 64
population  = 10
generations = 10
threshold   = 0.0050

def serve_model(epochs, units1, act1, units2, act2, classes, act3, loss, opt, IP_Train, OP_Train, summary=False):
    model = Sequential()
    model.add(LSTM(units1,input_shape=(IP_Train.shape[1], IP_Train.shape[2]),return_sequences=True))
    model.add(Activation(act1))
    model.add(LSTM(units2,return_sequences=False))
    model.add(Activation(act2))
    model.add(Dense(classes))
    model.add(Activation(act3))
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    
    if summary:
        model.summary()

    model.fit(IP_Train, OP_Train, batch_size=batch_size, epochs=epochs, verbose=0)

    return model

class Network():
    def __init__(self):
        self._epochs = np.random.randint(1, 15)

        self._units1 = np.random.randint(1, 500)
        self._units2 = np.random.randint(1, 500)

        self._act1 = random.choice(['sigmoid', 'relu', 'softmax', 'tanh', 'elu', 'selu', 'linear'])
        self._act2 = random.choice(['sigmoid', 'relu', 'softmax', 'tanh', 'elu', 'selu', 'linear'])
        self._act3 = random.choice(['sigmoid', 'relu', 'softmax', 'tanh', 'elu', 'selu', 'linear'])

        self._loss = random.choice([
            'categorical_crossentropy',
            'binary_crossentropy',
            'mean_squared_error',
            'mean_absolute_error',
            'sparse_categorical_crossentropy'
        ])
        self._opt = random.choice(['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'])

        self._accuracy = 0

    def init_hyperparams(self):
        hyperparams = {
            'epochs': self._epochs,
            'units1': self._units1,
            'act1'  : self._act1,
            'units2': self._units2,
            'act2'  : self._act2,
            'act3'  : self._act3,
            'loss'  : self._loss,
            'optimizer': self._opt
        }
        return hyperparams

def init_networks(population):
    return [Network() for _ in range(population)]

def fitness(networks):
    for network in networks:
        hyperparams = network.init_hyperparams()
        epochs = hyperparams['epochs']
        units1 = hyperparams['units1']
        act1   = hyperparams['act1']
        units2 = hyperparams['units2']
        act2   = hyperparams['act2']
        act3   = hyperparams['act3']
        loss   = hyperparams['loss']
        opt    = hyperparams['optimizer']

        try:
            model = serve_model(epochs, units1, act1, units2, act2, classes, act3, loss, opt, IP_Train, OP_Train)
            accuracy = model.evaluate(IP_Test, OP_Test, verbose=0)[1]
            network._accuracy = accuracy
            print ('Accuracy: {}'.format(network._accuracy))
        except:
            network._accuracy = 0
            print ('Build failed.')

    return networks

def selection(networks):
    networks = sorted(networks, key=lambda network: network._accuracy, reverse=True)
    networks = networks[:int(0.2 * len(networks))]

    return networks

def crossover(networks):
    offspring = []
    for _ in range(int((population - len(networks)) / 2)):
        parent1 = random.choice(networks)
        parent2 = random.choice(networks)
        child1 = Network()
        child2 = Network()

        # Crossing over parent hyper-params
        child1._epochs = int(parent1._epochs/4) + int(parent2._epochs/2)
        child2._epochs = int(parent1._epochs/2) + int(parent2._epochs/4)

        child1._units1 = int(parent1._units1/4) + int(parent2._units1/2)
        child2._units1 = int(parent1._units1/2) + int(parent2._units1/4)

        child1._units2 = int(parent1._units2/4) + int(parent2._units2/2)
        child2._units2 = int(parent1._units2/2) + int(parent2._units2/4)

        child1._act1 = parent2._act2
        child2._act1 = parent1._act2

        child1._act2 = parent2._act1
        child2._act2 = parent1._act1

        child1._act3 = parent2._act2
        child2._act3 = parent1._act2

        offspring.append(child1)
        offspring.append(child2)

    networks.extend(offspring)

    return networks

def mutate(networks):
    for network in networks:
        if np.random.uniform(0, 1) <= 0.1:
            network._epochs += np.random.randint(0,100)
            network._units1 += np.random.randint(0,100)
            network._units2 += np.random.randint(0,100)

    return networks

def main():
    networks = init_networks(population)

    for gen in range(generations):
        print ('Generation {}'.format(gen+1))

        networks = fitness(networks)
        networks = selection(networks)
        networks = crossover(networks)
        networks = mutate(networks)

        for network in networks:
            if network._accuracy > threshold:
                print ('Threshold met')
                print (network.init_hyperparams())
                print ('Best accuracy: {}'.format(network._accuracy))
                exit(0)

if __name__ == '__main__':
    main()    
#%%