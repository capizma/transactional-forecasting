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

parameters = [{'changepoint_prior_scale': [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5],},
{'seasonality_prior_scale': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],},
{'holidays_prior_scale' : [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],},
{'seasonality_mode' : ['additive','multiplicative'],}]

if __name__ == '__main__':
    for idx,param_set in enumerate(parameters):
      
      # Generate all combinations of parameters
      all_params = [dict(zip(param_set.keys(), v)) for v in itertools.product(*param_set.values())]
      mapes = []  # Store the RMSEs for each params here
    
      ls = ['Faster_Payment_Inbound_Retail','Faster_Payment_Outbound_Retail','Retail_BACS_Direct_Debit','Retail_BACS_Direct_Credit','Retail Debit Card ATM Us On Us']
    
      tuning_results = pd.DataFrame(all_params)
      
      for idx2,item in enumerate(ls):
        df = pd.read_csv('../data/'+item+'.csv')
        df = df[['reporting_date','volume']]
        df = df.sort_values(by='reporting_date')
        df.columns = ['ds','y']
    
        df = regressors.produce_flags(df[['ds','y']])
        df = df.drop(['monday','friday'],axis=1)
        
        mapes = []
    
        for params in all_params:
    
          m = Prophet(**params)
    
          m.add_country_holidays(country_name='UK')
    
          for col in df.columns[2:]:
            m.add_regressor(col)
    
          m.fit(df)
    
          initial = str(len(df)-(30*7)) + 'days'
    
          df_cv = cross_validation(m,initial=initial, period = '30 days', horizon='30 days', parallel="processes")
          
          df_p = performance_metrics(df_cv, rolling_window=1)
          mapes.append((df_p['smape'].values[0]*100))
    
        # Find the best parameters
        tuning_results['mape_set'+str(idx2)] = mapes
    
      all_results.append(tuning_results)
    
    def plotthing(data):
      fig,ax = plt.subplots()
      
      for i in range(1,len(data.columns)):
        sns.lineplot(x=data.index.values, y=data[data.columns[i]].values)
      
      # Set the ytick locations and labels, can also use np array here
      ax.set_xticklabels([0]+[str(x) for x in data[data.columns[0]].values])
      ax.set_xlabel(data.columns[0])
      ax.set_ylabel("sMAPE (%)")
      
      # Show the plot
      plt.show()
    
    for result in all_results[:-1]:
      plotthing(result)
      