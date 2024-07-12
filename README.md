# Energy Consumption Prediction using RNN and LSTM

This Jupyter Notebook demonstrates how to use Recurrent Neural Networks (RNN) and Long Short-Term Memory networks (LSTM) to predict energy consumption.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
    1. [Importing Libraries](#1-importing-libraries)
    2. [Data Loading and Initial Exploration](#2-data-loading-and-initial-exploration)
    3. [Data Preparation](#3-data-preparation)
    4. [Data Processing](#4-data-processing)
    5. [Model Building and Training](#5-model-building-and-training)
        1. [RNN Model](#rnn-model)
        2. [LSTM Model](#lstm-model)
    6. [Evaluation](#6-evaluation)
        1. [Prepare Test Data](#prepare-test-data)
        2. [RNN Predictions](#rnn-predictions)
        3. [LSTM Predictions](#lstm-predictions)
6. [Results](#results)


## Introduction
This project aims to predict future energy consumption using time series data. We employ machine learning techniques, specifically RNN and LSTM models, to achieve this.

## Dataset
The dataset used in this project is `energy_consumption.csv`, which contains time-stamped energy consumption data.

## Installation
To run this notebook, you need to have the following libraries installed:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras
```
## Usage
1. Clone the repository.
2. Ensure you have the dataset energy_consumption.csv in the same directory as the notebook.
3. Run the Jupyter Notebook to execute the code cells step-by-step.
## Project Structure
The notebook is divided into the following sections:

 ### 1. **Importing Libraries**

   Essential libraries such as NumPy, Pandas, Matplotlib, Scikit-learn, and Keras are imported.

```python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, Dense, Dropout
```
### 2. **Data Loading and Initial Exploration**
The dataset is loaded and initial exploration is performed.

```python
df = pd.read_csv('energy_consumption.csv', index_col=['Datetime'], parse_dates=['Datetime'])
df.head()
df.info()
```
### 3. **Data Preparation**
The data is split into training and test sets, and scaled using `MinMaxScaler`.

```python
train_size = int(len(df) * 0.8)
train = df[0:train_size]
test = df[train_size:]

scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train)
```
### 4. **Data Processing**
The data is reshaped into a format suitable for training RNN and LSTM models.

```python

time_step = 50
x_train, y_train = [], []

for i in range(time_step, len(scaled_train)):
    x_train.append(scaled_train[i-time_step:i, 0])
    y_train.append(scaled_train[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
```
### 5. **Model Building and Training**
Both RNN and LSTM models are built, compiled, and trained.

#### RNN Model
```python

rnn_model=Sequential()

rnn_model.add(SimpleRNN(64,activation='tanh',return_sequences=True,input_shape=(x_train.shape[1],1)))
rnn_model.add(Dropout(0.2))

rnn_model.add(SimpleRNN(64,activation='tanh',return_sequences=True))
rnn_model.add(Dropout(0.20))

rnn_model.add(SimpleRNN(64,activation='tanh',return_sequences=True))
rnn_model.add(Dropout(0.20))

rnn_model.add(SimpleRNN(64,activation='tanh',return_sequences=True))
rnn_model.add(Dropout(0.20))

rnn_model.add(SimpleRNN(64,activation='tanh',return_sequences=True))
rnn_model.add(Dropout(0.20))

rnn_model.add(SimpleRNN(64))
rnn_model.add(Dropout(0.20))

rnn_model.add(Dense(1))

rnn_model.compile(optimizer='adam', loss='mse')
rnn_model.fit(x_train, y_train, epochs=5)
```
#### LSTM Model
```python
rnn_model=Sequential()

rnn_model.add(SimpleRNN(64,activation='tanh',return_sequences=True,input_shape=(x_train.shape[1],1)))
rnn_model.add(Dropout(0.2))

rnn_model.add(SimpleRNN(64,activation='tanh',return_sequences=True))
rnn_model.add(Dropout(0.20))

rnn_model.add(SimpleRNN(64,activation='tanh',return_sequences=True))
rnn_model.add(Dropout(0.20))

rnn_model.add(SimpleRNN(64,activation='tanh',return_sequences=True))
rnn_model.add(Dropout(0.20))

rnn_model.add(SimpleRNN(64,activation='tanh',return_sequences=True))
rnn_model.add(Dropout(0.20))

rnn_model.add(SimpleRNN(64))
rnn_model.add(Dropout(0.20))

rnn_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(x_train, y_train, epochs=5)
```
### 6. **Evaluation**
The models are evaluated on the test set, and the results are visualized.

#### Prepare Test Data
```python
test_array = test.values
total_data = pd.concat((df['DOM_MW'], test['DOM_MW']), axis=0)
inputs = total_data[len(total_data) - len(test) - time_step:].values.reshape(-1, 1)
inputs = scaler.transform(inputs)

x_test = []
for i in range(time_step, inputs.shape[0]):
    x_test.append(inputs[i-time_step:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
```
#### **RNN Predictions**
```python
ypred = rnn_model.predict(x_test)
ypred = scaler.inverse_transform(ypred)

plt.figure(figsize=(15, 8))
plt.plot(test_array, color='red', label='Actual Energy')
plt.plot(ypred, color='blue', label='Predicted Energy')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Actual Energy & Predicted Energy')
plt.legend()
plt.show()
```
![energy RNN](https://github.com/user-attachments/assets/dbb245c5-67dc-4fc1-a00b-2337d7edac27)

#### **LSTM Predictions**
```python
y_pred = lstm_model.predict(x_test)
y_pred = scaler.inverse_transform(y_pred)

plt.figure(figsize=(15, 8))
plt.plot(test_array, color='red', label='Actual Energy')
plt.plot(y_pred, color='blue', label='Predicted Energy')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Actual Energy & Predicted Energy')
plt.legend()
plt.show()
```

![Energy LSTM](https://github.com/user-attachments/assets/643f5277-5576-42ac-95ad-3aa2a6c8a3f7)

## Results
The notebook will output model performance metrics and visualizations that compare the predicted values against the actual energy consumption values.



