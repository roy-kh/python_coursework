# Roy Hayyat
# ITP 359 2023
# Homework 2

"""Compare a feed forward neural network and a hidden markov model to forecast a time series"""

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dateutil.relativedelta import relativedelta

'''1. Download global temperature anomaly from 1850 to the present. Temperatures should be monthly.
   2. Read the data file'''
myDF = pd.read_csv("/content/temperature.csv")
print(myDF)

'''3. Convert the Year column into datetime (from pandas)'''
myDF['Year'] = pd.to_datetime(myDF['Year'])
print(myDF)

'''4. Plot the temperature anomaly vs year'''
plt.figure()
plt.plot(myDF['Year'], myDF['Anomaly'])
plt.xlabel('Year')
plt.ylabel('Temp Anomaly')
plt.show()

'''5. Save only the temperature anomaly into a 2D numpy array'''
anomaly_array = myDF['Anomaly'].values.reshape(-1, 1)
print(anomaly_array)

'''6. Scale the temperature using minmaxscaler'''
scaler = MinMaxScaler()
temp_scaled = scaler.fit_transform(anomaly_array)
print(temp_scaled)

'''7. Convert temp array into sequences of n monthly temps that are shifted by one month in each row of the array. 
# Call this array X. y is the next month’s temp (1D array). The length of the sequence n is your choice (e.g. 24).'''
def to_sequences(temp_scaled, seq_size):
    x = []
    y = []
    for i in range(len(temp_scaled)- seq_size - 1):
        window = temp_scaled[i:(i+seq_size), 0]
        x.append(window)
        y.append(temp_scaled[i+seq_size, 0])
    return np.array(x), np.array(y)

seq_size = 24  # Number of previous time steps to use as input variables
X, y = to_sequences(temp_scaled, seq_size)

print('Shape of data set: ', X.shape)
print(X)
print('Shape of y: ', y.shape)
print(y)

'''8. Build a dense feedforward neural network with sequences of temperature as input and y as the output.'''
model = Sequential()
model.add(Input(shape=(seq_size,)))  # only need to define the input shape
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())

'''9. Use mean squared error as the loss function'''
model.compile(loss='mean_squared_error', optimizer='adam')

'''10. Train the network'''
model_fit = model.fit(X, y, verbose=1, epochs=20)

'''11. Predict the next months temperature for all sequences in X. Inverse scale the temperature.'''
y_pred_nn = model.predict(X)
y_pred_nn = scaler.inverse_transform(y_pred_nn)

'''12. Score the model'''
score_nn = mean_squared_error(scaler.inverse_transform(y.reshape(-1, 1)), y_pred_nn[:, 0])
print('Neural Network MSE:', score_nn)

'''13. Now, plot the predicted temperatures and the actual temperatures'''
plt.figure()
plt.plot(myDF['Year'], scaler.inverse_transform(temp_scaled))
plt.plot(myDF['Year'][-len(y_pred_nn):], y_pred_nn)  # predicted
plt.title('Neural Network Predictions')
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly')

'''14. Now predict the temperature for the next 24 months into the future. 
   For every month in the future, you will predict the temperature. 
   Each month's predicted temperature should be stored back into the sequence for predicting the following month. 
   So, each y value becomes part of X.'''
print(X.shape)
adjusted_X = X.copy()  # keep a copy of X

X = np.append(X, np.zeros((seq_size, seq_size)), axis=0)  # zeros act as placeholders for predictions
# post current month
z = []

for i in range(len(X) - seq_size - 1, len(X) - 1):  # used as month input, to find prediction of upcoming month
  calculated_prediction = model.predict(X[[i]])
  for j in range(1,24):  # we use this to shift to the right, finding a place to put the prediction
    X[i + 1, j - 1] = X[i, j]
  X[i + 1, j] = calculated_prediction
  z.append(myDF.iloc[i, 0] + relativedelta(months = 1 + seq_size))

plt.figure()
plt.plot(myDF['Year'], scaler.inverse_transform(temp_scaled), label='Actual Anomaly')
plt.plot(myDF['Year'][-len(y_pred_nn):], y_pred_nn)
plt.plot(z, scaler.inverse_transform(X[-1, :].reshape(-1, 1)))
plt.show()
