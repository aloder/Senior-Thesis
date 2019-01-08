#%%
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

features = ['temp', 'work']
data = pd.read_csv('i1.csv')
data.date = pd.to_datetime(data.date)
test = data.set_index('date')['2017'].reset_index()
training = data.set_index('date')[:'2016'].reset_index()

trainingD = (training[features].values, training['load'].values)
testD = (test[features].values, test['load'].values)

(x_train, y_train) = trainingD
(x_test, y_test) = testD
test

#%%
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(256, activation=tf. nn.relu,
                           input_shape=(x_train.shape[1],)),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mape'])
    return model
model = build_model()
model.summary()

#%%
EPOCHS = 100

# Store training stats
history = model.fit(x_train, y_train, epochs=EPOCHS,
                    validation_split=0.2, verbose=False)

#%%
import matplotlib.pyplot as plt
def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_percentage_error']),
           label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_percentage_error']),
           label = 'Val loss')
    plt.legend()

plot_history(history)

#%%
[loss,mpe] = model.evaluate(x_test, y_test, verbose=0)
print("Testing set Mean percent Error: {:7.2f}".format(mpe))


#%%
test_predictions = model.predict(x_test).flatten()
plt.scatter(y_test, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])


#%%
error = test_predictions - y_test
plt.hist(error, bins = 50)
plt.xlabel("Prediction Error [1000$]")
_ = plt.ylabel("Count")