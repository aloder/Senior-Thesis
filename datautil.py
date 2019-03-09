import numpy as np
import pandas as pd
import tensorflow as tf

def readData(path):
    data = pd.read_csv(path)
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

def getData(path='data_out/data_formatted_cleaned2.csv'):
    data = readData(path)
    data = data.merge(lastHourValues(data), on='date')
    data = dayMean(data)
    data = dayOfWeek(data)
    data = cyclicalTime(data)
    data.reset_index(inplace=True)
    return data

def reorder(x_data, y_data):
    order = np.argsort(np.random.random(y_data.shape))
    x_data = x_data[order]
    y_data = y_data[order]
    return x_data, y_data

# TODO fix the static nature
def datasets(
    data,
    testDates= [['2017-2'], ['2017-5'], ['2017-9'], ['2017-11']],
    trainingDates = [['','2016'], ['2018'], ['2017-1'], ['2017-3', '2017-4'], ['2017-6', '2017-8'], ['2017-10'], ['2017-12']],
    tbText=[]):

    tbText.append(lambda: tf.summary.text('Test Dates', tf.convert_to_tensor(str(testDates))))

    tbText.append(lambda: tf.summary.text('Training Dates', tf.convert_to_tensor(str(trainingDates))))
    data = data.set_index('date')
    test = data['2017-2'].append(data['2017-5']).append(data['2017-9']).append(data['2017-11']).reset_index()
    training = data[:'2016'].append(data['2018']).append(data['2017-1']).append(data['2017-3':'2017-4']).append(data['2017-6':'2017-8']).append(data['2017-10']).append(data['2017-12']).reset_index()
    return  training, test

def normalize(features, data):
    for f in features:
        mean = data[f].mean(axis=0)
        std = data[f].std(axis=0)
        data[f] = (data[f] - mean) / std
    return data