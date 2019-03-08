import tensorflow as tf
import pandas as pd
import os
import sys

import matplotlib.pyplot as plt

def loadModels(path):
    model = os.path.join(sys.argv[1], "models")
    bestModelPath = os.path.join(model, "bestTrain.hdf5")
    finalModelPath = os.path.join(model, "final.hdf5")

    bestModel = tf.keras.models.load_model(bestModelPath)
    finalModel = tf.keras.models.load_model(finalModelPath)
    return bestModel, finalModel

def readInput():
    if len(sys.argv) == 1: 
        l = os.listdir("logs")
        print("pick log file:")
        for index, d in enumerate(l):
            print("[{}] {}".format(index, d))
        inp = input("")
        return 'logs/'+l[int(inp)]

    return sys.argv[1]



def loadTrainingAndTestCSV(path):
    csv = os.path.join(path, "csv")
    trainingPath = os.path.join(csv, "training.csv")
    testPath = os.path.join(csv, "test.csv")

    training = pd.read_csv(trainingPath)
    training.date = pd.to_datetime(training.date)

    test = pd.read_csv(testPath)
    test.date = pd.to_datetime(test.date)
    return training, test

def createFigures(training, test):
    test.plot(x='load_prediction', y='load', kind='scatter', title="test scatterplot")
    plt.figure(1)

    training.plot(x='date', y=['load','load_prediction', 'dif'], legend=True, title="training load prediction")
    plt.figure(2)


    test.plot(x='date', y=['load','load_prediction'], legend=True, title="test load prediction")
    plt.figure(3)

    plt.show()

path = readInput()
training, test = loadTrainingAndTestCSV(path)

training['dif'] = abs(training['load'] - training['load_prediction'])
test['dif'] = abs(test['load'] - test['load_prediction'])

createFigures(training,test)