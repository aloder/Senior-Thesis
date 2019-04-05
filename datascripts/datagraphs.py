import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('data_out/data_formatted_cleaned2.csv')
data.date = pd.to_datetime(data.date)
data = data.set_index('date')

loadDay = data['2017-05-16 11':'2017-05-17 11']
loadDay.plot(y='temperature', legend=True)
plt.figure(1)

loadDay.plot(y='load', legend=True)
plt.figure(2)


plt.show()