
#%%
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)



#%%
data = pd.read_csv('data_out/data_formatted_02.csv')
data.date = pd.to_datetime(data.date)

# add last hour values to data
data2 = data[['date', 'load','temperature', 'cloud_cover']]
data2.date = pd.to_datetime(data2.date.map(lambda x: x - pd.DateOffset(hours=-1)))
data2 = data2.rename(columns={'load':'last_hour_load', 'temperature':'last_hour_temperature', 'cloud_cover': 'last_hour_cloud_cover'})
data = data.merge(data2, on='date')

data = data.set_index('date')
#%%
data['day_load_mean'] = data.load.rolling(24).mean()
data['day_temperature_mean'] = data.temperature.rolling(24).mean()
features = ['temperature', 'hour', 'work', 'cloud_cover' ]
features += ['last_day_load', 'last_week_load', 'last_year_load']
features += ['day_load_mean', 'day_temperature_mean']
features += ['last_hour_cloud_cover', 'last_hour_load', 'last_hour_temperature']

## typical is all data up to 2018 is training and 2018 is test
test = data['2017-2'].append(data['2017-5']).append(data['2017-9']).append(data['2017-11']).reset_index()
training = data[:'2016'].append(data['2018']).append(data['2017-1']).append(data['2017-3':'2017-4']).append(data['2017-6':'2017-8']).append(data['2017-10']).append(data['2017-12']).reset_index()


#%%
training = training[features+['load']].dropna()
test2 = test[features+['load']].dropna()
x_train, y_train = training[features].values, training['load'].values
x_test, y_test = test2[features].values, test2['load'].values

#%%
# normalize the features - better training

mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std
order = np.argsort(np.random.random(y_train.shape))
x_train = x_train[order]
y_train = y_train[order]
#%%
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(1024,
            input_shape=(x_train.shape[1],), activation='relu'),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])

    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mape'])
    return model
model = build_model()
model.summary()


#%%
modelSave = 'models/weights-val_5.hdf5'
EPOCHS = 10000

earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=500)
save = tf.keras.callbacks.ModelCheckpoint(modelSave, monitor='val_loss', save_best_only=True)
history = model.fit(x_train, y_train, epochs=EPOCHS,
                    validation_split=0.2, 
                    batch_size=100, verbose=2, callbacks=[earlyStop, save])

#%%
[loss,mpe] = model.evaluate(x_test, y_test, verbose=0)

print("Testing set Mean Abs percent Error: {:7.2f}".format(mpe))
model2 = tf.keras.models.load_model(modelSave)
[loss2,mpe2] = model2.evaluate(x_test, y_test, verbose=0)
print("Testing set Mean Abs best percent Error: {:7.2f}".format(mpe2))
#%%
import matplotlib.pyplot as plt
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