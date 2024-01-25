## Time Series - Forecasting Notes
git remote add origin https://github.com/theCodeCS/kaggle_competitions.git
git branch -M main
git push -u origin main

### Goals

- engineer features to model the major time series components (trends, seasons, and cycles),
- visualize time series with many kinds of time series plots,
- create forecasting hybrids that combine the strengths of complementary models, and
- adapt machine learning methods to a variety of forecasting tasks.

### Lesson 1: Time Series Basics

#### What is a Time Series?

A time series is a sequence of measurements of the same variable collected over time. Most often, measurements are made at regular intervals. In forecasting applications, the observations are typically recorded with a regular frequency, like daily or monthly.

#### Time Step Features

1. **Time Step Features**: These are numeric features that capture the time steps between observations in the data. For example, if the data is recorded daily, then the time step feature for the second observation would be 1 (day), for the third observation 2 (days), and so on.

**Example**
```python
import numpy as np

df['Time'] = np.arange(len(df.index))

df.head()
```

2. **Lag Features**: These features are the observations at previous time steps. For example, if the data is recorded daily, the lag feature for the second observation would be the first observation, for the third observation it would be the second observation, and so on.

Time step features can be directly derived frm the time index. Lag features can be created using the `shift()` method.
The most basic time-step feature is the time dummy, which counts off time steps in the series from beginning to end. The time dummy is a simple way to capture the trend in the data.

Lag features allow us to model **Serial Dependence**(which is the tendency of a time series observation to be dependent on previous observations. Serial dependence is also called autocorrelation.)

**Example**
```python
df['Lag_1'] = df['Hardcover'].shift(1)
df = df.reindex(columns=['Hardcover', 'Lag_1'])

df.head()
```

#### Feature Engineering

A large part of adapting machine learning to time series problems is largely about feature engineering with the time index and lags. The goal is to create features that capture the most important information from the time series data and that are directly related to the forecasting task.
