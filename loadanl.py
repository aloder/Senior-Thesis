import pandas as pd

data = pd.read_csv('anly/anltraining.csv')
data.date = pd.to_datetime(data.date)

import matplotlib.pyplot as plt

data['dif'] = abs(data['load'] - data['load_prediction'])

data.plot(x='date', y=['load','load_prediction', 'dif'])
plt.show()