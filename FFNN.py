import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from time import time
import datetime
import argparse

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

tbText = []
def get_file_path(dpath, tag, ext='csv', join = 'csv'):
    file_name = tag + '.'+ext
    folder_path = os.path.join(dpath, join)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, file_name)


def readData():
    data = pd.read_csv('data_out/data_formatted_cleaned2.csv')
    data.date = pd.to_datetime(data.date)
    return data




# add last hour values to data
def lastHourValues(data):
    data2 = data[['date', 'load','temperature', 'cloud_cover']]
    data2.date = pd.to_datetime(data2.date.map(lambda x: x - pd.DateOffset(hours=-1)))
    data2 = data2.rename(columns={'load':'last_hour_load', 'temperature':'last_hour_temperature', 'cloud_cover': 'last_hour_cloud_cover'})
    return data2

def twoHourLoad(data):
    data2 = data[['date', 'load']]
    data2.date = pd.to_datetime(data2.date.map(lambda x: x - pd.DateOffset(hours=-2)))
    data2 = data2.rename(columns={'load':'2_hour_load'})
    return data2

def dayMean(data):
    data['day_load_mean'] = data.load.rolling(24).mean()
    data['day_temperature_mean'] = data.temperature.rolling(24).mean()
    return data

def dayOfWeek(data):
    data['day_of_week'] = data.date.map(lambda x: x.dayofweek)
    return data

def cyclicalTime(data):
    data = data.set_index('date')
    data['sin_time'] = np.sin(2*np.pi*data.hour/24)
    data['cos_time'] = np.cos(2*np.pi*data.hour/24)
    data['sin_day'] = np.sin(2*np.pi*data.day_of_year/365)
    data['cos_day'] = np.cos(2*np.pi*data.day_of_year/365)
    data['sin_day_of_week'] = np.sin(2*np.pi*data.day_of_week/6)
    data['cos_day_of_week'] = np.cos(2*np.pi*data.day_of_week/6)
    return data

def getData():
    data = readData()
    data = data.merge(lastHourValues(data), on='date')
    data = dayMean(data)
    data = dayOfWeek(data)
    data = cyclicalTime(data)
    return data



def getFeatures():
    features = ['temperature', 'year', 'work',  'cloud_cover' ]
    features += ['sin_day', 'cos_day','cos_time', 'sin_time']
    features += ['sin_day_of_week', 'cos_day_of_week']
    features += ['last_day_load',  'last_week_load']
    features += ['day_load_mean', 'day_temperature_mean']
    features += ['last_hour_load', 'last_hour_temperature']
    features += ['last_hour_temperature']
    tbText.append(tf.summary.text('Features', tf.convert_to_tensor(str(features))))
    return features



## typical is all data up to 2018 is training and 2018 is test
def datasets(data):
    testDates = [['2017-2'], ['2017-5'], ['2017-9'], ['2017-11']]
    trainingDates = [['','2016'], ['2018'], ['2017-1'], ['2017-3', '2017-4'], ['2017-6', '2017-8'], ['2017-10'], ['2017-12']]

    tbText.append(tf.summary.text('Test Dates', tf.convert_to_tensor(str(testDates))))

    tbText.append(tf.summary.text('Training Dates', tf.convert_to_tensor(str(trainingDates))))

    test = data['2017-2'].append(data['2017-5']).append(data['2017-9']).append(data['2017-11']).reset_index()
    training = data[:'2016'].append(data['2018']).append(data['2017-1']).append(data['2017-3':'2017-4']).append(data['2017-6':'2017-8']).append(data['2017-10']).append(data['2017-12']).reset_index()
    return  training, test


def getXY(data, features):
    data = data[features+['load']].dropna()
    x, y = data[features].values, data['load'].values
    return x, y

def normalize(x_train, x_test):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    return x_train, x_test

def reorder(x_data, y_data):
    order = np.argsort(np.random.random(y_data.shape))
    x_data = x_data[order]
    y_data = y_data[order]
    return x_data, y_data

def getAllXYs(training, test, features):
    x_train, y_train = getXY(training, features)
    x_test, y_test = getXY(test, features)

    x_train, x_test = normalize(x_train, x_test)
    x_train_ordered = x_train
    x_train, y_train = reorder(x_train, y_train)
    return x_train, y_train, x_test, y_test, x_train_ordered

def build_model(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(1024,
            input_shape=input_shape, activation='relu'),
        keras.layers.Dense(2048, activation='relu'),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1)
    ])

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
    tensorboard = TensorBoard(log_dir=STORE_PATH)

    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience)
    modelSave = get_file_path(STORE_PATH, 'bestTrain', 'hdf5', 'models')
    save = tf.keras.callbacks.ModelCheckpoint(modelSave, monitor='val_loss', save_best_only=True)
    model.fit(x_train, y_train, epochs=EPOCHS,
                        validation_split=validation_split, 
                        batch_size=batch_size, verbose=verbose, callbacks=[earlyStop, save, tensorboard])
    bestModel = tf.keras.models.load_model(modelSave)

    modelSave2 = get_file_path(STORE_PATH, 'final', 'hdf5', 'models')
    tf.keras.models.save_model(model, modelSave2)

    return model, bestModel

def trainingTestingLoss(model, x_test, y_test, typeStr=""):
    [loss,mpe] = model.evaluate(x_test, y_test, verbose=0)
    trainingLoss = "[{}] Testing set Mean Abs percent Error: {:7.2f}".format(typeStr, mpe)
    print(trainingLoss)
    tbText.append(tf.summary.text('Testing Loss: {}'.format(typeStr), tf.convert_to_tensor(trainingLoss)))

def generatePredictions(model, data, x_data, features):
    predictions = model.predict(x_data).flatten()

    data = data[features+['load', 'date']].dropna()
    data['load_prediction'] = predictions
    return data

def runModel(
    STORE_PATH,
    model_func=build_model,
    EPOCHS= 10000, 
    patience = 30, 
    batch_size= 5000,
    verbose=0,
    validation_split= 0.2
    ):
    configText = "EPOCHS={}, patience={}, batch_size={}, verbose={}, validation_split={}".format(EPOCHS, patience, batch_size, verbose, validation_split)
    tbText.append(tf.summary.text('Config', tf.convert_to_tensor(configText)))
    data = getData()
    training, test = datasets(data)
    features = getFeatures()
    x_train, y_train, x_test, y_test, x_train_ordered = getAllXYs(training, test, features)
    model = model_func((x_train.shape[1],))

    tbText.append(tf.summary.text('Layers', tf.convert_to_tensor(str(model.layers))))
    model, bestModel = trainModel(model, x_train, y_train, STORE_PATH, EPOCHS=EPOCHS, patience=patience, batch_size=batch_size, verbose=verbose, validation_split=validation_split)

    trainingTestingLoss(model, x_test, y_test, "Final Model")
    trainingTestingLoss(bestModel, x_test, y_test, "Best Model")

    test = generatePredictions(bestModel, test, x_test, features)
    test.to_csv(get_file_path(STORE_PATH,'test', 'csv', 'csv'))
    training = generatePredictions(bestModel, training, x_train_ordered, features)
    training.to_csv(get_file_path(STORE_PATH, 'training', 'csv', 'csv'))

def configureSession():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    session = tf.Session(config=config)
    return session

def storePathMetaData(
    test = 'none',
    typeOfNetwork= 'FFNN',
    version = '0.01',
    note= ""): 
    today = datetime.datetime.now()
    
    todayStr = today.strftime("%m_%d_%Y_%H_%M_%S")
    tbText.append(tf.summary.text('Date', tf.convert_to_tensor(todayStr)))
    modelName = '{}-{}-v{}-{}'.format(todayStr,typeOfNetwork, version, test)

    tbText.append(tf.summary.text('Version-test', tf.convert_to_tensor("v{}-{}".format(version, test))))
    tbText.append(tf.summary.text('Note', tf.convert_to_tensor(note)))

    STORE_PATH = "logs/{}".format(modelName)
    return STORE_PATH

def writeText(session, tbText, STORE_PATH):
    summary_writer = tf.summary.FileWriter(STORE_PATH)
    for index, summary_op in enumerate(tbText):
        text = session.run(summary_op)
        summary_writer.add_summary(text, index)
    summary_writer.close()

def main(args, model_func=build_model):
    session = configureSession()
    STORE_PATH = storePathMetaData(test=args.test, version=args.version, note=args.note)
    runModel(STORE_PATH, model_func=model_func, EPOCHS=args.epochs, patience=args.patience, batch_size=args.batch_size, verbose=args.verbose, validation_split=args.validation_split)
    writeText(session, tbText, STORE_PATH)
    session.close()


def model1(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(1024,
            input_shape=input_shape, activation='relu'),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1)
    ])

    model.compile(loss='mse',
    optimizer='adam',
    metrics=['mape'])
    return model

def model2(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(2048,
            input_shape=input_shape, activation='relu'),
        keras.layers.Dense(2048, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1)
    ])
    model.compile(loss='mse',
    optimizer='adam',
    metrics=['mape'])
    return model

def model3(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(2048,
            input_shape=input_shape, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(2048, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(loss='mse',
    optimizer='adam',
    metrics=['mape'])
    return model
testModels = [
    [
        model1,
        "LotsOfBigLayers"
    ],
    [
        model2,
        "twoBigLayers"
    ],
    [
        model3,
        "twoBigLayersPastDropout"
    ] 
]

args=parser.parse_args()
for mt in testModels:
    tbText = []
    [model, test] = mt
    args.test = test
    main(args, model)