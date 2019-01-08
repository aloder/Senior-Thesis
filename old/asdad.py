
#%%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, CuDNNLSTM, Dropout
import numpy as np
import pandas as pd


#%%
data = pd.read_csv('input2.csv')
data.work = data.work.map(lambda x: int(x))
data2 = data.drop(['pload'], axis=1)
data2.date = pd.to_datetime(data2.date)
data2['doy'] = data2.date.map(lambda x: x.dayofyear)
data2['hour'] = data2.date.map(lambda x: x.hour)
data2


#%%
data3  = data2[['date', 'load']]
data3.date = pd.to_datetime(data3.date.map(lambda x: x - pd.DateOffset(years=-1)))
data3.rename(columns={'load':'lyload'}, inplace=True)
data3


#%%
datalagged = data2.merge(data3, left_on='date', right_on='date')
datalagged


#%%
test = data2.set_index('date')['2017'].reset_index()
training = data2.set_index('date')[:'2016'].reset_index()
features = ['temp', 'doy', 'work', 'hour', 'load']


#%%
training[features].values


#%%
def to_sequences(seq_size, obs):
    x = []
    y = []

    for i in range(len(obs)-SEQUENCE_SIZE-1):
        window = obs[i:(i+SEQUENCE_SIZE)]
        after_window = obs[i+SEQUENCE_SIZE][len(features) - 1]
        window = [x for x in window]
        #print("{} - {}".format(window,after_window))
        x.append(window)
        y.append(after_window)
        
    return np.array(x),np.array(y)
    
    
SEQUENCE_SIZE = 25
x_train,y_train = to_sequences(SEQUENCE_SIZE, training[features].values)
x_test,y_test = to_sequences(SEQUENCE_SIZE, test[features].values)


#%%
print("Shape of training set: {}".format(x_train.shape))
print("Shape of test set: {}".format(x_test.shape))
x_train


#%%
def build_model2():
    model = keras.Sequential([
        CuDNNLSTM(256, input_shape=x_train.shape[1:], return_sequences=True),
        Dropout(0.2),
        CuDNNLSTM(256),
        Dropout(0.2),
        Dense(256,activation='relu'),
        Dropout(0.2),
        Dense(256,activation='relu'),
        Dense(128,activation='relu'),
        Dense(1)
    ])
    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mape'])
    return model
model = build_model2()
model.summary()


#%%
x_train


#%%
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=60)

EPOCHS = 500

# Store training stats
history = model.fit(x_train, y_train, epochs=EPOCHS,
                    validation_split=0.2)


#%%
import matplotlib.pyplot as plt


def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_percentage_error']),
           label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_percentage_error']),
           label = 'Val loss')
    plt.legend()

plot_history(history)


#%%
[loss,mpe] = model.evaluate(x_test, y_test, verbose=0)
print("Testing set Mean Abs percent Error: {:7.2f}".format(mpe))


#%%



