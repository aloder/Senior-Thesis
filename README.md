# Senior Thesis


## About

This is a senior thesis for Trinity University

### Working abstract
Load forecasting greatly influences the energy production to meet the forecasted demand. If the forecasted demand is off this could lead to blackout or excess waste of precious resources. This paper compares Feed-forward Deep Neural Networks (FF-DNN), different models of Recurrent Deep Neural Networks (R-DNN) specifically Long short Term Memory (LSTM), and looks into combining the two approaches into an ensemble network. We will be predicting load with an hourly granularity also known as Short Term Load Forecasting (STLF). We will be applying these approaches to real world data sets from \url{www.eia.gov} over a period of about 4 years. Our approach will be focused on the integration of historical time features from the last hour, day, month, etc. with the inclusion of R-DNN methods. We show that the included time features reduce the overall error and increase generalizability. We combine this with features such as weather, cyclical time features, cloud cover, and the day of year to further reduce the error. We will then compare the approaches to reveal that the correct handling of time features significantly improves the model by learning hidden features. 

## Usage

To get help using:
```
python FFNN.py -h
```

We use tensorboard for the output.

Navigate to directory and then:
```
tensorboard --logdir=logs\
```

everything is saved in the log files, the models and all.

## Paper

Coming soon

## A Good Readme

Coming eventually