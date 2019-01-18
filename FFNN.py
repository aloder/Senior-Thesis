
#%%
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from time import time

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)



#%%
data = pd.read_csv('data_out/data_formatted_cleaned2.csv')
data.date = pd.to_datetime(data.date)

# add last hour values to data
data2 = data[['date', 'load','temperature', 'cloud_cover']]
data2.date = pd.to_datetime(data2.date.map(lambda x: x - pd.DateOffset(hours=-1)))
data2 = data2.rename(columns={'load':'last_hour_load', 'temperature':'last_hour_temperature', 'cloud_cover': 'last_hour_cloud_cover'})
data = data.merge(data2, on='date')
data2 = data[['date', 'load']]
data2.date = pd.to_datetime(data2.date.map(lambda x: x - pd.DateOffset(hours=-2)))
data2 = data2.rename(columns={'load':'2_hour_load'})
data = data.merge(data2, on='date')
data['day_of_week'] = data.date.map(lambda x: x.dayofweek)
data = data.set_index('date')
data['sin_time'] = np.sin(2*np.pi*data.hour/24)
data['cos_time'] = np.cos(2*np.pi*data.hour/24)
data['sin_day'] = np.sin(2*np.pi*data.day_of_year/365)
data['cos_day'] = np.cos(2*np.pi*data.day_of_year/365)
data['sin_day_of_week'] = np.sin(2*np.pi*data.day_of_week/6)
data['cos_day_of_week'] = np.cos(2*np.pi*data.day_of_week/6)
#%%
data['day_load_mean'] = data.load.rolling(24).mean()
data['day_temperature_mean'] = data.temperature.rolling(24).mean()
features = ['temperature', 'year', 'work',  'cloud_cover' ]
features += ['sin_day', 'cos_day','cos_time', 'sin_time']
features += ['sin_day_of_week', 'cos_day_of_week']
features += ['last_day_load',  'last_week_load']
features += ['day_load_mean', 'day_temperature_mean']
features += ['last_hour_load', 'last_hour_temperature']
features += ['last_hour_temperature']

## typical is all data up to 2018 is training and 2018 is test
test = data['2017-2'].append(data['2017-5']).append(data['2017-9']).append(data['2017-11']).reset_index()
training = data[:'2016'].append(data['2018']).append(data['2017-1']).append(data['2017-3':'2017-4']).append(data['2017-6':'2017-8']).append(data['2017-10']).append(data['2017-12']).reset_index()
#test = data['2017'].reset_index()
#training = data[:'2016'].append(data['2018']).reset_index()

#%%
training2 = training[features+['load']].dropna()
test2 = test[features+['load']].dropna()
x_train, y_train = training2[features].values, training2['load'].values
x_test, y_test = test2[features].values, test2['load'].values

#%%
# normalize the features - better training

mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std
x_train2 = x_train
order = np.argsort(np.random.random(y_train.shape))
x_train = x_train[order]
y_train = y_train[order]
#%%
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(2048,
            input_shape=(x_train.shape[1],), activation='relu'),
        #keras.layers.Dense(2048, activation='relu'),
        #keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1)
    ])

    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mape'])
    return model
model = build_model()
model.summary()


#%%
modelSave = 'models/weights-day-ahead.hdf5'
EPOCHS = 10000
patience = 100
batch_size = 5000

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience)
save = tf.keras.callbacks.ModelCheckpoint(modelSave, monitor='val_loss', save_best_only=True)
history = model.fit(x_train, y_train, epochs=EPOCHS,
                    validation_split=0.2, 
                    batch_size=batch_size, verbose=2, callbacks=[earlyStop, save, tensorboard])

#%%
[loss,mpe] = model.evaluate(x_test, y_test, verbose=0)
print("Testing set Mean Abs percent Error: {:7.2f}".format(mpe))

model2 = tf.keras.models.load_model(modelSave)
[loss2,mpe2] = model2.evaluate(x_test, y_test, verbose=0)
print("Testing set Mean Abs best percent Error: {:7.2f}".format(mpe2))


#%%

def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(history.epoch[100:], np.array(history.history['mean_absolute_percentage_error'][100:]),
           label='Train Loss')
    plt.plot(history.epoch[100:], np.array(history.history['val_mean_absolute_percentage_error'][100:]),
           label = 'Val loss')
    plt.legend()
    plt.show()
plot_history(history)

#%%
test_predictions = model.predict(x_test).flatten()
plt.scatter(y_test, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
plt.plot([-100, 100], [-100, 100])
plt.show()

test = test[features+['load', 'date']].dropna()
test['load_prediction'] = test_predictions
test.plot(x='date', y=['load','load_prediction'], legend=True)
plt.show()
test.to_csv('anltestdayahead.csv')


training_predictions = model2.predict(x_train2).flatten()
training = training[features+['load', 'date']].dropna()
training['load_prediction'] = training_predictions
training.plot(x='date', y=['load','load_prediction'], legend=True)
plt.show()

training.to_csv('anltrainingdayahead.csv')