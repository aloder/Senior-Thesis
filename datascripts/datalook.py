import pandas as pd

data = pd.read_csv('data_out/data_formatted_04.csv')
data.date = pd.to_datetime(data.date)

import matplotlib.pyplot as plt


# range '2016-11-06 22':'2016-11-09 12
# 2018-04-11 02
# 2018-05-04 08?
# 2017-06-14 22
# 2016-12-13 16
# 2016-12-09 21
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

data.plot(y=['load'], legend=True)

data.to_csv('data_out/data_formatted_cleaned4.csv')
plt.show()
