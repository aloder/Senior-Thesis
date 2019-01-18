
#%%
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)



#%%
data = pd.read_csv('data_out/data_formatted_cleaned2.csv')
data.date = pd.to_datetime(data.date)
data['day_of_week'] = data.date.map(lambda x: x.dayofweek)
data['load_actual'] = data['load']
data = data.set_index('date')


data['sin_time'] = np.sin(2*np.pi*data.hour/24)
data['cos_time'] = np.cos(2*np.pi*data.hour/24)
data['sin_day'] = np.sin(2*np.pi*data.day_of_year/365)
data['cos_day'] = np.cos(2*np.pi*data.day_of_year/365)
data['sin_day_of_week'] = np.sin(2*np.pi*data.day_of_week/6)
data['cos_day_of_week'] = np.cos(2*np.pi*data.day_of_week/6)

#%%
features = [ 'load','temperature','sin_time', 'cos_time', 'work']
features2 = ['work', 'last_week_load', 'last_day_load', 'temperature', 'cloud_cover']
features2 += ['sin_time', 'cos_time', 'sin_day', 'cos_day', 'sin_day_of_week', 'cos_day_of_week']

def normalize(features, data):
    for f in features:
        mean = data[f].mean(axis=0)
        std = data[f].std(axis=0)
        data[f] = (data[f] - mean) / std
    return data
data = data.reset_index()
featureList = list(set(features+features2))
data = normalize(featureList, data[featureList+['load_actual', 'date']].dropna())
data = data.set_index('date')
# 'hour', 'work',  'last_year_load', 'cloud_cover',  'last_week_load', 
## typical is all data up to 2018 is training and 2018 is test
test = data['2017-2'].append(data['2017-5']).append(data['2017-9']).append(data['2017-11']).reset_index()
training = data[:'2016'].append(data['2018']).append(data['2017-1']).append(data['2017-3':'2017-4']).append(data['2017-6':'2017-8']).append(data['2017-10']).append(data['2017-12']).reset_index()
#test = data['2018'].reset_index()
#training = data[:'2017'].reset_index()
#%%
def to_sequences(seq_size, obs):
    x = []
    z = []
    y = []
    win = obs[features].values
    awin = obs['load_actual'].values
    zwin = obs[features2].values
    for i in range(len(obs)-SEQUENCE_SIZE-1):
        
        window = win[i:(i+SEQUENCE_SIZE)]
        after_window = awin[i+SEQUENCE_SIZE]
        window = [x for x in window]
        #print("{} - {}".format(window,after_window))
        x.append(window)
        y.append(after_window)
        z.append(zwin[i+SEQUENCE_SIZE])
        
    return np.array(x), np.array(z), np.array(y)


SEQUENCE_SIZE = 25
x_train,z_train,y_train = to_sequences(SEQUENCE_SIZE, training)
x_test,z_test,y_test = to_sequences(SEQUENCE_SIZE, test)

#%%
x_train

#%%
order = np.argsort(np.random.random(y_train.shape)) 
x_train = x_train[order]
y_train = y_train[order]

#%%
def build_model():
    model = keras.Sequential()
    input1 = keras.layers.Input(shape=x_train.shape[1:])
    input2 = keras.layers.Input(shape=z_train.shape[1:])
    model1_out = keras.layers.CuDNNLSTM(512)(keras.layers.CuDNNLSTM(512, return_sequences=True)(input1))
    model2_out = keras.layers.Dense(256, activation='relu')(input2)

    concat = keras.layers.concatenate([model1_out, model2_out])
    dense = keras.layers.Dense(256, activation='relu')(concat)
    drop = keras.layers.Dropout(0.025)(dense)
    out = keras.layers.Dense(1)(drop)
    model = keras.models.Model(inputs=[input1, input2], outputs=out)
    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mape'])
    return model
model = build_model()
model.summary()

#%%
modelSave = 'models/weights-LSTM_3.hdf5'
EPOCHS = 10000

earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=200)
save = tf.keras.callbacks.ModelCheckpoint(modelSave, monitor='val_loss', save_best_only=True)
history = model.fit([x_train,z_train], y_train, epochs=EPOCHS,
                    validation_split=0.2, 
                    batch_size=5000, verbose=2, callbacks=[earlyStop, save])

#%%
[loss,mpe] = model.evaluate([x_test,z_test], y_test, verbose=0)

print("Testing set Mean Abs percent Error: {:7.2f}".format(mpe))
model2 = tf.keras.models.load_model(modelSave)
[loss2,mpe2] = model2.evaluate([x_test,z_test], y_test, verbose=0)
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
test_predictions = model.predict([x_test, z_test]).flatten()
plt.scatter(y_test, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
plt.plot([-100, 100], [-100, 100])
plt.show()

print(test_predictions.shape)
print(y_test.shape)
test = test[features+['load_actual', 'date']].dropna()
test = test.iloc[len(test)-len(test_predictions):]
test['loadpre'] = test_predictions
test['err'] = abs(test['load_actual'] - test['loadpre'])
test.plot(x='date', y=['load_actual','loadpre', 'err'], legend=True)
plt.show()
res = pd.DataFrame({'pre':test_predictions, 'act':y_test})
res.plot(y=['pre', 'act'], legend=True)
plt.show()