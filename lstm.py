import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# from subprocess import check_output
# print check_output(['ls', '../ConEdison']).decode('utf8')

import warnings
warnings.filterwarnings('ignore')
from time import time

start = time()
# ==========Initial trade parameters =============
symbol = 'BTC/USD'
timeframe = '1d'
since = '2017-01-01 00:00:00'

header = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']

# ==========Initial exchange parameters =============
# kraken = exchange_data('kraken', 'BTC/USD', timeframe=timeframe, since=hist_start_date)
# write_to_csv(kraken,'BTC/USD','kraken')

# data = pd.DataFrame(kraken, columns=header)
# print(data.head())
data = pd.read_csv("gemini_BTCUSD_1min.csv")

# train_data = data['Close'][:11000]
# test_data = data['Close'][11000:]
# Scale data to be between 0 and 1
# Remember normalise test and training data wrt to training data
scl = MinMaxScaler()

'''
train_data =  train_data.reshape(-1, 1)
test_data = train_data.reshape(-1, 1)

# reshape both train and test data

train_data = train_data.reshape(-1)

# Normalise test data
test_data = scaler.transform(test_data).reshape(-1)
'''

# Scale the data
cl = data['Close'].values.reshape(data['Close'].shape[0],1)
cl = scl.fit_transform(cl)

# Create a function to process the data into N day lookback slioes
def processData(data,lookback):
    X,Y = [],[]
    for i in range(len(data)-lookback-1):
        X.append(data[i:(i+lookback),0])
        Y.append(data[(i+lookback),0])
    return np.array(X), np.array(Y)

X,y = processData(cl, 7)

X_train,X_test = X[:int(X.shape[0]*0.8)], X[int(X.shape[0]*0.8):]
y_train,y_test = y[:int(y.shape[0]*0.80)],y[int(y.shape[0]*0.80):]
print(X_train.shape[0])
print(X_test.shape[0])
print(y_train.shape[0])
print(y_test.shape[0])

# LSTM Model - 5 essential components
# Cell State - this represents the internal memory of the cell
# which stores both short term as well as long term

# hidden state - This is output state information calculated wrt
# current input, previous hidden state and current cell input

# input gate - decides how much infomation from the ccurrent input flows to cell state

# Forget state - decides how much information flows from the current input and previous
# cell state

# output gate

model = Sequential
model.add(LSTM(500, input_shape=(7,1)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')

# reshape data for (Sample, Timestamp, Features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_train.shape[0], X_train.shape[1], 1)

# Fit model and check for overfitting

history = model.fit(X_train,y_train,epochs=300,
                    validation_data=(X_test,y_test),
                    shuffle=False)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

Xt = model.predict(X_test)
plt.plot(scl.inverse_transform(y_test.reshape(-1,1)))
plt.plot(scl.inverse_transform(Xt))

actual = []
predicted = []
i = 1500
Xt = model.predict(X_test[i].reshape(1,7,1))

print('Predicted:{0}, Actual:{1}'.format(scl.inverse_transform(Xt),
                                         scl.inverse_transform(y_test[i].reshape(-1,1))))

predicted.append(scl.inverse_transform(Xt))
actual.append(scl.inverse_transform(y_test[i].reshape(-1,1)))

result_df = pd.DataFrame({'Predicted':list(np.reshape(predicted,-1)),
                          'Actual':list(np.reshape(actual,-1))})



