
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
data = data.set_index('date')
data['load_actual'] = data['load']
#%%
features = ['temperature', 'load']
# 'hour', 'work',  'last_year_load', 'cloud_cover',  'last_week_load', 
## typical is all data up to 2018 is training and 2018 is test
test = data['2018'].reset_index()
training = data[:'2017'].reset_index()


#%%
def to_sequences(seq_size, obs):
    x = []
    y = []

    for i in range(len(obs)-SEQUENCE_SIZE-1):
        window = obs[i:(i+SEQUENCE_SIZE)]
        after_window = obs[i+SEQUENCE_SIZE][len(features)]
        window = [x[:-1] for x in window]
        #print("{} - {}".format(window,after_window))
        x.append(window)
        y.append(after_window)
        
    return np.array(x),np.array(y)

def normalize(features, data):
    for f in features:
        mean = data[f].mean(axis=0)
        std = data[f].std(axis=0)
        data[f] = (data - mean) / std
    return data
# normalize everything except the labels (load_actual)
training2 = normalize(features, training[features + [ 'load_actual']].dropna())
test2 = normalize(features, test[features + [ 'load_actual']].dropna())

SEQUENCE_SIZE = 48
x_train,y_train = to_sequences(SEQUENCE_SIZE, training2.values)
x_test,y_test = to_sequences(SEQUENCE_SIZE, test2.values)


#%%
x_train

#%%
order = np.argsort(np.random.random(y_train.shape))
x_train = x_train[order]
y_train = y_train[order]

#%%
def build_model():
    model = keras.Sequential([
        keras.layers.CuDNNLSTM(256, input_shape=x_train.shape[1:]),
        keras.layers.Activation('relu'),
        keras.layers.Dense(512),
        keras.layers.Dense(512),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])

    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mape'])
    return model
model = build_model()
model.summary()
x_train.shape

#%%
modelSave = 'models/weights-LSTM_3.hdf5'
EPOCHS = 10000

earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=30)
save = tf.keras.callbacks.ModelCheckpoint(modelSave, monitor='val_loss', save_best_only=True)
history = model.fit(x_train, y_train, epochs=EPOCHS,
                    validation_split=0.2, 
                    batch_size=2000, verbose=2, callbacks=[earlyStop, save])

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
test.merge(test_predictions)
test.plot(x='date', y=['load','loadpre'], legend=True)
plt.show()