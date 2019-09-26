# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from params import params
#%%
train = False
#%%
dataset = pd.read_csv('Google_Stock_Price_Train.csv',index_col="Date",parse_dates=True)
#%
dataset.head()

dataset.describe()

dataset.columns

dataset.isna().any()

dataset.info()

dataset['Open'].plot(figsize=(16,6))

#dataset['Close'] = pd.to_numeric(dataset.Close)

# convert column "a" of a DataFrame
dataset["Close"] = dataset["Close"].str.replace(',', '').astype(float)

dataset["Volume"] = dataset["Volume"].str.replace(',', '').astype(float)

# 7 day rolling mean
dataset.rolling(7).mean().head(20)

dataset['Open'].plot(figsize=(16,6))
dataset.rolling(window=30).mean()['Close'].plot()

dataset['Close: 30 Day Mean'] = dataset['Close'].rolling(window=30).mean()
dataset[['Close','Close: 30 Day Mean']].plot(figsize=(16,6))

# Optional specify a minimum number of periods
dataset['Close'].expanding(min_periods=1).mean().plot(figsize=(16,6))

training_set=dataset['Open']
training_set=pd.DataFrame(training_set)

#%%
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv',index_col="Date",parse_dates=True)

real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_test.head()

dataset_test.info()

dataset.columns

dataset_test["Volume"] = dataset_test["Volume"].str.replace(',', '').astype(float)

test_set=dataset_test['Open']

test_set=pd.DataFrame(test_set)

test_set.info()

#%%
# Feature Scaling
sc                  = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
#%%
# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train = np.array(X_train)
y_train = np.array(y_train)
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#%%
# Part 2 - Building the RNN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.layers import LSTM
from keras.layers import Dropout
from keras import Model
from params import params

#%%
def model():
    Input_lstm  = Input(shape=(X_train.shape[1],1))

    L1          = LSTM(50,return_sequences=True)(Input_lstm)
    D1          = Dropout(0.2)(L1)

    L2          = LSTM(50,return_sequences=True)(D1)
    D2          = Dropout(0.2)(L2)

    L3          = LSTM(50,return_sequences=True)(D2)
    D3          = Dropout(0.2)(L3)

    L4          = LSTM(50,return_sequences=True)(D3)
    D4          = Dropout(0.2)(L4)

    Out         = LSTM(1)(D4)

    model       = Model(inputs=Input_lstm,outputs=Out)
    model.compile(optimizer='adam',loss='mean_squared_error')
    return model

def model_(x):
    layer       = []
    Drop        = []

    Input_lstm  = Input(shape=(X_train.shape[1],1))

    L1          = LSTM(x['unit'],return_sequences=True)(Input_lstm)
    D1          = Dropout(x['drop'])(L1)

    layer.append(L1)
    Drop.append(D1)

    for ii in range(x['layer_no']):
        L = LSTM(x['unit'],return_sequences=True)(Drop[ii])
        layer.append(L)
        D = Dropout(x['drop'])(layer[ii+1])
        Drop.append(D)
    
    Out         = LSTM(1)(Drop[ii+1])
    
    model       = Model(inputs=Input_lstm,outputs=Out)
    model.compile(optimizer='adam',loss='mean_squared_error')
    return model

#%%
parameters = {}

list_unit  = [50,100,150]
list_drop  = [0.15,0.30]
list_layer = [3,5,7]
list_epoch = [150,300]
list_batch = [32,64]

if train:
    for unit in list_unit:
        for drop in list_drop:
            for layer in list_layer:
                for epoch in list_epoch:
                    for batch in list_batch:
                        parameters['unit']      = unit
                        parameters['drop']      = drop
                        parameters['layer_no']  = layer
                        parameters['epoch']     = epoch
                        parameters['batch']     = batch
                        lstm_model = model_(parameters)
                        history    = lstm_model.fit(X_train,y_train,epochs=parameters['epoch'],
                                                    batch_size=parameters['batch'])
                        parameters['loss'] = min(history.history['loss'])
                        #%%
                        # Part 3 - Making the predictions and visualising the results
                        # Getting the predicted stock price of 2017
                        dataset_total = pd.concat((dataset['Open'], dataset_test['Open']), axis = 0)
                        inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
                        inputs = inputs.reshape(-1,1)
                        inputs = sc.transform(inputs)
                        X_test = []
                        for i in range(60, 80):
                            X_test.append(inputs[i-60:i, 0])
                        X_test = np.array(X_test)
                        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                        predicted_stock_price = lstm_model.predict(X_test)
                        predicted_stock_price = sc.inverse_transform(predicted_stock_price)

                        #%%
                        predicted_stock_price=pd.DataFrame(predicted_stock_price)
                        predicted_stock_price.info()

                        #%%
                        # Visualising the results
                        # plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
                        # plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
                        # plt.title('Google Stock Price Prediction')
                        # plt.xlabel('Time')
                        # plt.ylabel('Google Stock Price')
                        # plt.legend()
                        # plt.show()

                        #%%
                        params(experiment_name='exp_stock',parameters=parameters,filename='params.txt')
else:
    parameters['unit']      = 50
    parameters['drop']      = 0.15
    parameters['layer_no']  = 3
    parameters['epoch']     = 300
    parameters['batch']     = 32
    lstm_model = model_(parameters)
    history    = lstm_model.fit(X_train,y_train,epochs=parameters['epoch'],
                                batch_size=parameters['batch'])
    parameters['loss'] = min(history.history['loss'])
    #
    # Part 3 - Making the predictions and visualising the results
    dataset_total = pd.concat((dataset['Open'], dataset_test['Open']), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, 80):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = lstm_model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    #%%
    predicted_stock_price=pd.DataFrame(predicted_stock_price)
    predicted_stock_price.info()

    #%%
    # Visualising the results
    plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
    plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
    plt.title('Google Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Google Stock Price')
    plt.legend()
    plt.show()

#%%


#%%
