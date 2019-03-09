import pandas as pd
import numpy as np
import tensorflow as tf
import datetime
import os


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
    note= "",
    tbText = []): 
    today = datetime.datetime.now()
    
    todayStr = today.strftime("%m_%d_%Y_%H_%M_%S")
    tbText.append(lambda: tf.summary.text('Date', tf.convert_to_tensor(todayStr)))
    modelName = '{}-{}-v{}-{}'.format(todayStr,typeOfNetwork, version, test)

    tbText.append(lambda: tf.summary.text('Version-test', tf.convert_to_tensor("v{}-{}".format(version, test))))
    tbText.append(lambda: tf.summary.text('Note', tf.convert_to_tensor(note)))

    STORE_PATH = "logs/{}".format(modelName)
    return STORE_PATH

def writeText(session, tbText, STORE_PATH):
    summary_writer = tf.summary.FileWriter(STORE_PATH)
    for index, func in enumerate(tbText):
        summary_op = func()
        text = session.run(summary_op)
        summary_writer.add_summary(text, index)
    tbText.clear()

    summary_writer.close()

def get_file_path(dpath, tag, ext='csv', join = 'csv'):
    file_name = tag + '.'+ext
    folder_path = os.path.join(dpath, join)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, file_name)

def generatePredictions(model, data, x_data, features):
    predictions = model.predict(x_data).flatten()

    data = data[features+['load', 'date']].dropna()
    data['load_prediction'] = predictions
    return data

def trainingTestingLoss(model, x_test, y_test, typeStr=""):
    [loss,mpe] = model.evaluate(x_test, y_test, verbose=0)
    trainingLoss = "[{}] Testing set Mean Abs percent Error: {:7.2f}".format(typeStr, mpe)
    return [trainingLoss, loss, mpe]