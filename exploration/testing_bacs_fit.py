import os
from prophet import Prophet
import pandas as pd
import datetime
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
from utils import regressors
from sklearn.metrics import mean_absolute_error
import itertools
import itertools
import logging
import json
import warnings
import tqdm
import json
import numpy as np
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
import seaborn as sns
sns.set(style="darkgrid")

all_results = []

ls = ['Retail_BACS_Direct_Credit.csv']

if __name__ == '__main__':
  
    for idx2,item in enumerate(ls):
        df = pd.read_csv('../data/'+item)
        
        df = df[['reporting_date','volume']]
        df = df.sort_values(by='reporting_date')
        df.columns = ['ds','y']
    
        df = regressors.produce_flags(df[['ds','y']])
        df = df[['ds','y','business_end_lf','business_end_lf_lag']]
    
        mapes = []
    
        m = Prophet()
    
        m.add_country_holidays(country_name='UK')
    
        for col in df.columns[2:]:
          m.add_regressor(col)
    
        m.fit(df)
    
        initial = str(len(df)-(30*7)) + 'days'
    
        df_cv = cross_validation(m,initial=initial, period = '30 days', horizon='30 days', parallel="processes")
    
        df_p = performance_metrics(df_cv, rolling_window=1)
    
    
    def plotthing(data):
      fig,ax = plt.subplots()
      
      for i in ['yhat','y']:
        sns.lineplot(x=data.index.values, y=data[i].values)
      
      # Set the ytick locations and labels, can also use np array here
      #ax.set_xticklabels([0]+[str(x) for x in data[data.columns[0]].values])
      #ax.set_xlabel(data.columns[0])
      #ax.set_ylabel("MAPE (%)")
      
      # Show the plot
      plt.show()
    
    plotthing(df_cv)
    
    m = Prophet(yearly_seasonality=True)
    
    m.add_country_holidays(country_name='UK')
    
    for col in df.columns[2:]:
      m.add_regressor(col)
    
    m.fit(df.iloc[:-30])
    
    results = m.predict(df)
    
    results['y'] = df['y']
    
    plotthing(results) 