#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from datetime import date
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import pytz
import os 
cwd = os.getcwd()
print(cwd)
path = 'Senior-Thesis/'
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar(),  normalize=True)

def isWorkingDay(x):
    d = us_bd.rollback(date(x.year, x.month, x.day))
    return d.day == x.day and d.month == x.month and d.year == x.year

#%%
with open(path+'data/tusonload.json') as f:
    datafile = json.load(f)
ldf = pd.DataFrame(datafile['series'][0]['data'], columns=['date', 'load'])
ldf.date = pd.to_datetime(ldf.date)


#%%
a = ldf['load'].isnull()
b = a.cumsum()
c = ldf['load'].bfill()
d = c + (b-b.mask(a).bfill().fillna(0).astype(int)).sub(1)
ldf['load'] =  ldf['load'].fillna(d)
ldf

#%%
# convert day to 1 or zero
ldf.set_index(['date'], inplace=True)
ldf.sort_index(inplace=True)

# remove bad data
ldf = ldf[ldf.load < 10000][:'2018-07']
ldf



weatherdf = pd.read_fwf(path+'data/tusonweather.txt')
weatherdf

#%%
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
wdf.date = pd.to_datetime(wdf['date'], format='%Y%m%d%H%M')
wdf.date = wdf.date.dt.round('H')
wdf['TEMP'] = wdf[wdf.TEMP != '****'].TEMP.transform(lambda x: int(x))
wdf['DEWP'] = wdf[wdf.DEWP != '****'].DEWP.transform(lambda x: int(x))
wdf['SKC'] = wdf[wdf.SKC != '***'].SKC.transform(skytonum)
wdf.fillna(method='ffill', inplace=True)
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
adf
#%%
data = adf
dropDays = [
    '2018-04-11', '2018-05-04', '2017-06-14', '2016-12-13', 
    '2016-12-09', '2016-10-01', '2016-10-20', '2018-01-26', 
    '2016-12-28', '2016-10-18', '2017-01-06', '2016-09-30']
# 2018-01-26 23
# 2016-10-18 07
#data.drop(data['2016-11-06':'2016-11-09']).drop(data[dropDays]).plot(x='date', y=['load'], legend=True)
data = data.set_index('date')
data.drop(data['2016-11-06':'2016-11-09'].index, inplace=True)
data.drop(data['2015-09-14':'2015-09-17'].index, inplace=True)


for e in dropDays:
    data.drop(data[e].index, inplace=True)

adf = data.reset_index()

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
fdf.date = fdf.date.dt.tz_localize('UTC').dt.tz_convert('US/Mountain')

fdf['work'] = fdf['date'].map(lambda x: int(isWorkingDay(x)))

#%%
fdf.to_csv(path+'data_out/data_formatted_04.csv')
