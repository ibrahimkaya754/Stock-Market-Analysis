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
dataset = pd.read_csv('Google_Stock_Price_Train.csv',index_col="Date",parse_dates=True)
#%%
dataset.head()
#%%
dataset.describe()
#%%
dataset.columns
#%%
dataset.isna().any()
#%%
dataset.info()
#%%
dataset['Open'].plot(figsize=(16,6))
#%%
# The error here is about to have ',' inside some of the numeric values
#dataset['Close'] = pd.to_numeric(dataset.Close)
#%%
# convert column "a" of a DataFrame
dataset["Close"] = dataset["Close"].str.replace(',', '').astype(float)
#%%
dataset["Volume"] = dataset["Volume"].str.replace(',', '').astype(float)

#%%
# 7 day rolling mean
dataset.rolling(7).mean().head(20)
#%%
dataset['Open'].plot(figsize=(16,6))
dataset.rolling(window=30).mean()['Close'].plot()
#%%
dataset['Close: 30 Day Mean'] = dataset['Close'].rolling(window=30).mean()
dataset[['Close','Close: 30 Day Mean']].plot(figsize=(16,6))
#%%
# Optional specify a minimum number of periods
dataset['Close'].expanding(min_periods=1).mean().plot(figsize=(16,6))

#%%
training_set=dataset['Open']
training_set=pd.DataFrame(training_set)
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
print(X_train.shape)
#%%   VALIDATION / TEST SET
# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv',index_col="Date",parse_dates=True)
real_stock_price = dataset_test.iloc[:, 1:2].values
dataset_test.head()
dataset_test.info()
dataset.columns
dataset_test["Volume"] = dataset_test["Volume"].str.replace(',', '').astype(float)
test_set=dataset_test['Open']
test_set=pd.DataFrame(test_set)
test_set.info()
#%% Part 2 - Building the RNN
parameters     = {}
list_unit_size = [20,50,80,110]
list_dropout   = [0.1,0.15,0.20,0.25,0.30]
list_layer_no  = [1,2,3,4,5]
list_epoch     = [250,500,1000]
list_batch     = [16,32,64,128]
#%%
# Initialising the RNN
def model(parameters):
    Layer         = []
    
    Input_LSTM = Input(shape(X_train.shape[1],1))
    Layer.append(LSTM(parameters['unit_size'])(Input_LSTM)
    for layer_no in range(parameters['layer_no']):
        Layer





    regressor = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = parameters['unit_size'], return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(parameters['dropout']))

    for layer_no in range(int(parameters['layer_no'])):
        regressor.add(LSTM(units= parameters['unit_size'],return_sequences=True))
        regressor.add(Dropout(parameters['dropout']))

    # Adding the output layer
    regressor.add(Dense(units = 1))

    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    return regressor
#%%
exp_no = 1
for layer_no in list_layer_no:
    parameters['layer_no'] = layer_no
    for dropout in list_dropout:
        parameters['dropout'] = dropout
        for unit_size in list_unit_size:
            parameters['unit_size'] = unit_size
            for epoch in list_epoch:
                parameters['epoch'] = epoch
                for batch in list_batch:
                    parameters['batch'] = batch
                    model_LSTM = model(parameters)   
                    print(X_train.shape)
                    # Fitting the RNN to the Training set
                    model_LSTM.fit(X_train, y_train, epochs = parameters['epoch'], batch_size = parameters['batch'])         
                    
                    # Part 3 - Making the predictions and visualising the results

                    #%%
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
                    predicted_stock_price = model_LSTM.predict(X_test)
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
                    params(experiment_name='exp_lstm'+str(exp_no),parameters=parameters,filename = 'params_stock_market_lstm.txt')

                    #%%



#%%



