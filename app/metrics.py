import pandas as pd
import numpy as np
from numpy.polynomial import polynomial
import matplotlib.pyplot as plt

def generate_metrics(df):

  quartiles = [np.quantile(df['y'],0.25),np.quantile(df['y'],0.75)]
  iqr = quartiles[1] - quartiles[0]

  bounds = [quartiles[0]-(1.5*iqr),quartiles[1]+(1.5*iqr)]

  df_clean = df[(df['y'] > bounds[0]) & (df['y'] < bounds[1])]

  mean_with_outliers = round(np.mean(df['y']),2)
  mean_without_outliers = round(np.mean(df_clean['y']),2)

  max_volume = df['y'].max()
  max_volume_date = df['ds'].iloc[df['y'].idxmax()]

  min_volume = df['y'].min()
  min_volume_date = df['ds'].iloc[df['y'].idxmin()]

  stddev = round(df['y'].std(),2)

  c = np.polyfit(df.index.values, df['y'].values, 2)

  c = np.polyval(c, df.index.values)

  df['least_squares'] = c

  plt.plot(df.index.values,df[['y','least_squares']])

  initial = df.iloc[-90]['least_squares']
  final = df.iloc[-1]['least_squares']

  relative_change = lambda x,y : (y-x)/x

  trend = str(round(relative_change(initial,final)*100,2))+'%'

  return [mean_with_outliers,
         mean_without_outliers,
         max_volume,
         max_volume_date,
         min_volume,
         min_volume_date,
         stddev,
         trend]
