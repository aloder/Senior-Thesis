
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from datetime import date
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar(),  normalize=True)


#%%
def isWorkingDay(x):
    d = us_bd.rollback(date(x.year, x.month, x.day))
    return d.day == x.day and d.month == x.month and d.year == x.year
with open('data/tusonload.json') as f:
    datafile = json.load(f)
ldf = pd.DataFrame(datafile['series'][0]['data'], columns=['date', 'load'])
ldf.date = pd.to_datetime(ldf.date)

# convert day to 1 or zero
ldf['work'] = ldf['date'].map(lambda x: int(isWorkingDay(x)))
ldf.set_index(['date'], inplace=True)
ldf.sort_index(inplace=True)
# remove bad data
ldf = ldf[ldf.load < 10000][:'2018-07']
ldf


#%%
with open('data/tusonpredictedload.json') as f:
    datafile = json.load(f)
pldf = pd.DataFrame(datafile['series'][0]['data'], columns=['date', 'pload'])
pldf.rename(columns={'pload': 'day_ahead_load_prediction'}, inplace=True)
pldf.date = pd.to_datetime(pldf.date)
pldf.set_index(['date'], inplace=True)
pldf = pldf.sort_index()


#%%
weatherdf = pd.read_fwf('data/tusonweather.txt')
weatherdf

#%%
def fix(date):
    minutes = int(str(date)[-2:])
    ret = int(str(date)[:-2])
    if minutes > 30:
        ret += 1
    # Incorrect way to do it
    if (ret + 1)%25 == 0:
        ret -= 1
    return ret
def skytonum(sky): 
    if sky == 'CLR':
        return 0
    elif sky == 'SCT':
        return 1
    elif sky == 'BKN':
        return 2
    elif sky == 'OVC':
        return 3
    elif sky == 'OBS':
        return 4
    return None
wdf = weatherdf.get(['YR--MODAHRMN', 'TEMP','DEWP', 'SKC']).rename(columns={'YR--MODAHRMN': 'date'})
wdf['date'] = wdf['date'].transform(fix)
wdf['TEMP'] = wdf[wdf.TEMP != '****'].TEMP.transform(lambda x: int(x))
wdf['DEWP'] = wdf[wdf.DEWP != '****'].DEWP.transform(lambda x: int(x))
wdf['SKC'] = wdf[wdf.SKC != '***'].SKC.transform(skytonum)
wdf.dropna(thresh=2, inplace=True)
wdf.fillna(method='ffill', inplace=True)
wdf.date = pd.to_datetime(wdf['date'], format='%Y%m%d%H')
wdf = wdf.groupby('date').mean().reset_index()
wdf

#%%
wdf['year'] = wdf.date.map(lambda x: x.year)
wdf['week'] = wdf.date.map(lambda x: x.weekofyear)
wdf['day_of_year'] = wdf.date.map(lambda x: x.dayofyear)
wdf['hour'] = wdf.date.map(lambda x: x.hour)
wdf

#%%
adf = ldf.merge(wdf, on='date')
adf = adf.rename(columns={'TEMP': 'temperature', 'DEWP': 'dew_point', 'SKC': 'cloud_cover'})
adf = adf.merge(pldf, on='date')
adf

#%%
laggedyear  = adf[['date', 'load']]
laggedyear.date = pd.to_datetime(laggedyear.date.map(lambda x: x - pd.DateOffset(years=-1)))
laggedyear.rename(columns={'load':'last_year_load'}, inplace=True)

laggedweek  = adf[['date', 'load']]
laggedweek.date = pd.to_datetime(laggedweek.date.map(lambda x: x - pd.DateOffset(days=-7)))
laggedweek.rename(columns={'load':'last_week_load'}, inplace=True)

laggedday  = adf[['date', 'load']]
laggedday.date = pd.to_datetime(laggedday.date.map(lambda x: x - pd.DateOffset(days=-1)))
laggedday.rename(columns={'load':'last_day_load'}, inplace=True)

fdf = adf.merge(laggedyear,how='left',on='date').merge(laggedweek,how='left',on='date').merge(laggedday,how='left',on='date')

#%%
fdf.to_csv('data_out/data_formatted_02.csv')
