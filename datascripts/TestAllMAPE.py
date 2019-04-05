import tensorflow as tf
import pandas as pd
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    
def readInput():
    if len(sys.argv) == 1: 
        l = os.listdir("../logs")
        print("pick log file:")
        for index, d in enumerate(l):
            print("[{}] {}".format(index, d))
        inp = input("")
        return '../logs/'+l[int(inp)]

    return sys.argv[1]



def loadTrainingAndTestCSV(path):
    csv = os.path.join(path, "csv")
    trainingPath = os.path.join(csv, "training.csv")
    testPath = os.path.join(csv, "test.csv")
    if not os.path.exists(csv):
        return None
    training = pd.read_csv(trainingPath)
    training.date = pd.to_datetime(training.date)

    test = pd.read_csv(testPath)
    test.date = pd.to_datetime(test.date)
    return test

l = ['../logs/'+ f for f in os.listdir("../logs")]
for d in l:
    t = loadTrainingAndTestCSV(d)
    if t is None:
        continue
    mape = mean_absolute_percentage_error(t['load'].values, t['load_prediction'].values)
    print(d+" MAPE: "+ str(round(mape, 2)))