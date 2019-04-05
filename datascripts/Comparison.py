import tensorflow as tf
import pandas as pd
import os
import sys
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from dateutil import tz
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def loadModels(path):
    model = os.path.join(sys.argv[1], "models")
    bestModelPath = os.path.join(model, "bestTrain.hdf5")
    finalModelPath = os.path.join(model, "final.hdf5")

    bestModel = tf.keras.models.load_model(bestModelPath)
    finalModel = tf.keras.models.load_model(finalModelPath)
    return bestModel, finalModel

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

    training = pd.read_csv(trainingPath)
    training.date = pd.to_datetime(training.date)

    test = pd.read_csv(testPath)
    test.date = pd.to_datetime(test.date)
    return training, test

def createFigures(training, test):
    test.plot(x='load_prediction', y='load', kind='scatter', title="test scatterplot")
    plt.figure("Actual vs. Real")

    training.plot(x='date', y=['load','load_prediction', 'dif'], legend=True, title="training load prediction")
    plt.figure(2)


    test.plot(x='date', y=['load','load_prediction'], legend=True, title="test load prediction")
    plt.figure(3)

    testDate = test.set_index('date')
    hardDay = testDate['2017-05-16 11':'2017-05-17 11']
    hardDay.plot(y=['load','load_prediction'], legend=True, title="test load prediction hard day")
    plt.figure(4)

    plt.show()


def plotTemp(date):
    data = pd.read_csv('../data_out/data_formatted_03.csv')
    data.date = pd.to_datetime(data.date)
    tz2 = pytz.timezone('US/Mountain')
    data.date = data.date.dt.tz_localize(pytz.utc)
    data.date = data.date.dt.tz_convert(tz2)
    data = data.set_index('date')
    data = data[date]
    data = data.reset_index('date')

    ax = data.plot(x='date', y='temperature', grid=True, rot=0, title="Temperature "+date, legend=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%I %p', tz=tz.gettz('US/Mountain')))
    ax.set_xlabel('Hour')
    ax.set_ylabel('Temperature (F)')
    plt.show()

def plotCircle():
    data = pd.DataFrame(data={'hour': [x for x in range(24)]})
    data['sin_time'] = np.sin(2*np.pi*data.hour/24)
    data['cos_time'] = np.cos(2*np.pi*data.hour/24)
    ax = data.plot(x='sin_time', y='cos_time', kind='scatter', title="Cyclical Time Features")
    for i, txt in enumerate(data['hour']):
        ax.annotate(txt, (data['sin_time'].iloc[i],data['cos_time'].iloc[i]+0.03), size=10, xytext=(3,0), ha='right', textcoords='offset points')
    plt.show()

plotCircle()

paths = [
        ['../logs/03_18_2019_12_55_24-FFNN-v0.01-1BigLayersDropout2', 'FFNN'],
        ['../logs/03_28_2019_00_52_08-LSTM-v0.01-none', 'LSTM'],
        ['../logs/03_28_2019_21_24_52-merge-v0.01-relu6', 'Hybrid']
    ]
cur = None
plotTemp('2017-05-09')
for p in paths:
    training, test = loadTrainingAndTestCSV(p[0])

    test = test.drop_duplicates('date')
    a = [['2017-2', 'Winter'], ['2017-5','Spring'], ['2017-9','Summer'], ['2017-11', 'Fall']]

    print(p[1])        
    test = test.set_index('date')
    for m in a:
        data = test[m[0]].reset_index()
        err = mean_absolute_percentage_error(data['load'].values, data['load_prediction'].values)
        print(m[1]+": "+ str(round(err, 2)))
    test = test.reset_index()
    if cur is None:
        cur = test[['date','load','load_prediction']].rename(index=str, columns={'load_prediction': p[1]})
    else:
        cur = cur.merge(test[['date','load_prediction']].rename(index=str, columns={'load_prediction': p[1]}), on='date', validate="1:1")
    cur = cur.dropna()


exit

#harddays=['2017-05-16','2017-05-18', '2017-05-19']
#harddays=['2017-09-16']
#harddays=['2017-05-19']
#harddays=['2017-02-13']
#harddays=['2017-02-22']
harddays=['2017-11-21', '2017-02-06', '2017-05-09']
cur.date = cur.date.dt.tz_localize('UTC').dt.tz_convert('US/Mountain')

cur = cur.rename(index=str, columns={'load': 'Actual'})
ax = cur.plot( x='date', y=['FFNN', 'Hybrid', 'Actual'], grid=True, rot=0)
plt.show()
for day in harddays:
    cur2 = cur.set_index('date')
    cur2 = cur2[day]
    cur2 = cur2.reset_index()
    fig, axes = plt.subplots(nrows=2, ncols=3, sharey='row', squeeze=False, sharex=True,tight_layout=True, figsize=(10, 6), gridspec_kw={'height_ratios':[6, 1],'hspace': 0.0,'wspace': 0.0, 'top': 0.95, 'bottom': 0.1, 'left': 0.10, 'right': 0.90})
    fig.suptitle("Test Day "+ day)
    for i in range(len(paths)):
        a=paths[i][1]
        x=axes[0,i]
        ax = cur2.plot(ax=x, x='date', y=[a, 'Actual'], grid=True, rot=0)
        hours = mdates.HourLocator(interval = 4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        ax.xaxis.set_major_locator(hours)
        ax.set_ylabel('Load (MwH)')
        cur2['Absolute Error'] = abs(cur2['Actual'] - cur2[a])
        ax2 = cur2.plot(ax=axes[1,i], x='date', y=['Absolute Error'], legend=False, grid=True, rot=0)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

        ax2.text(0, 0, 'MAPE '+ str(mean_absolute_percentage_error(cur2['Actual'].values, cur2[a].values)), size=24)

        ax2.xaxis.set_major_locator(hours)
        ax2.set_ylabel('Error')
        ax2.set_xlabel('Hour\nMAPE '+ str(round(mean_absolute_percentage_error(cur2['Actual'].values, cur2[a].values), 2)))
    fig.tight_layout()
    if not os.path.exists("TestDay"+day+".png"):
        fig.savefig("TestDay"+day+".png")


