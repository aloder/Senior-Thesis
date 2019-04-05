import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from time import time
import datetime
import util
import datautil

defaultVersion = '0.01'

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')

parser.add_argument('-e','--epochs', type=int,
                    help='Amount of epochs to run', default=1000)
parser.add_argument('-p','--patience', type=int,
                    help='Amount of patience before ending the model training', default=30)

parser.add_argument('--validation_split', type=float,
                    help='Split the training set for validation', default=0.2)
parser.add_argument('--batch_size', type=int,
                    help='batch_size', default=5000)
parser.add_argument('-v','--verbose', type=int,
                    help='How verbose 0-3', default=0)
parser.add_argument('--version', type=str,
                    help='What version', default=defaultVersion)
parser.add_argument('--test', type=str,
                    help='What are you testing', default="none")
parser.add_argument('--note', type=str,
                    help='any notes', default="none")
parser.add_argument('--sequence_size', type=int,
                    help='size of the lstm', default=5)



def getFeatures():
    features = [ 'load_normalized', 'last_day_load', 'last_week_load','temperature', 'cloud_cover']
    return features

def build_model(shape):
    model = keras.Sequential()
    model.add(keras.layers.CuDNNLSTM(256, input_shape=shape, return_sequences=True))
    model.add(keras.layers.CuDNNLSTM(256))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1))

    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mape'])
    return model


def trainModel(
    model, 
    x_train, 
    y_train, 
    STORE_PATH, 
    EPOCHS= 10000, 
    patience = 30, 
    batch_size= 5000,
    verbose=0,
    validation_split= 0.2):
    tensorboard = TensorBoard(
        log_dir=STORE_PATH, 
        #histogram_freq=128,
        #write_grads=True, 
        #write_images=True
        )

    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience)
    modelSave = util.get_file_path(STORE_PATH, 'bestTrain', 'hdf5', 'models')
    save = tf.keras.callbacks.ModelCheckpoint(modelSave, monitor='val_loss', save_best_only=True)
    model.fit(x_train, y_train, epochs=EPOCHS,
                        validation_split=validation_split, 
                        batch_size=batch_size, verbose=verbose, callbacks=[earlyStop, save, tensorboard])
    bestModel = tf.keras.models.load_model(modelSave)

    modelSave2 = util.get_file_path(STORE_PATH, 'final', 'hdf5', 'models')
    tf.keras.models.save_model(model, modelSave2)

    return model, bestModel



def to_sequences(SEQUENCE_SIZE, features, obs):
    x = []
    y = []
    win = obs[features].values
    awin = obs['load'].values
    for i in range(len(obs)-SEQUENCE_SIZE):
        
        window = win[i:(i+SEQUENCE_SIZE)]
        after_window = awin[i+SEQUENCE_SIZE]
        window = [x for x in window]
        x.append(window)
        y.append(after_window)
        
    return np.array(x), np.array(y)


def runModel(
    STORE_PATH,
    model_func=build_model,
    EPOCHS= 10000, 
    patience = 30, 
    batch_size= 5000,
    verbose=0,
    validation_split= 0.2,
    SEQUENCE_SIZE=5,
    tbText = []
    ):
    configText = "EPOCHS={}, patience={}, batch_size={}, verbose={}, validation_split={}, SEQUENCE_SIZE={}".format(EPOCHS, patience, batch_size, verbose, validation_split, SEQUENCE_SIZE)
    tbText.append(lambda: tf.summary.text('Config', tf.convert_to_tensor(configText)))

    data = datautil.getData()
    data['load_normalized'] = data['load']
    features = getFeatures()
    tbText.append(lambda: tf.summary.text('Features', tf.convert_to_tensor(str(features))))
    data_normalized = datautil.normalize(features, data[features+['load', 'date']].dropna())

    training, test = datautil.datasets(data_normalized, tbText=tbText)

    trainingDropped= training.iloc[SEQUENCE_SIZE:]
    testDropped= test.iloc[SEQUENCE_SIZE:]

    x_train,y_train = to_sequences(SEQUENCE_SIZE, features, training)
    x_test,y_test = to_sequences(SEQUENCE_SIZE, features,  test)
    model = model_func(x_train.shape[1:])


    model, bestModel = trainModel(model, x_train, y_train, STORE_PATH, EPOCHS=EPOCHS, patience=patience, batch_size=batch_size, verbose=verbose, validation_split=validation_split)

    lossFinalStr = util.trainingTestingLoss(model, x_test, y_test, "Final Model")[0]
    print(lossFinalStr)
    tbText.append(lambda: tf.summary.text('Testing Loss: {}'.format("Final Model"), tf.convert_to_tensor(lossFinalStr)))

    lossStr = util.trainingTestingLoss(bestModel, x_test, y_test, "Best Model")[0]
    print(lossStr)
    tbText.append(lambda: tf.summary.text('Testing Loss: {}'.format("Best Model"), tf.convert_to_tensor(lossStr)))

    test = util.generatePredictions(bestModel, testDropped, x_test, features)
    test.to_csv(util.get_file_path(STORE_PATH,'test', 'csv', 'csv'))
    training = util.generatePredictions(bestModel, trainingDropped, x_train, features)
    training.to_csv(util.get_file_path(STORE_PATH, 'training', 'csv', 'csv'))


def main(args, model_func=build_model, run_model_func=runModel):
    tbText = []
    with util.configureSession() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        STORE_PATH = util.storePathMetaData(test=args.test, version=args.version, note=args.note, typeOfNetwork="LSTM", tbText=tbText)
        run_model_func(
            STORE_PATH, 
            tbText=tbText, 
            model_func=model_func, 
            EPOCHS=args.epochs, 
            patience=args.patience, 
            batch_size=args.batch_size, 
            verbose=args.verbose, 
            validation_split=args.validation_split, 
            SEQUENCE_SIZE=args.sequence_size)
        util.writeText(sess, tbText, STORE_PATH)
    tf.reset_default_graph()


if __name__ == "__main__":
    args=parser.parse_args()
    main(args)