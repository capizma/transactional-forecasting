import os
from fbprophet import Prophet
import pandas as pd
import datetime
from datetime import date, datetime, timedelta
import matplotlib
import matplotlib.pyplot as plt
from exploration.utils import regressors,forecast_utils
from sklearn.metrics import mean_absolute_error
import itertools
import itertools
import logging
import json
import warnings
import tqdm
import json
import numpy as np
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
import seaborn as sns
sns.set(style="darkgrid")

all_results = []

opts = ['holiday','monday','business_start','business_end']

regressor_list = ['baseline']
regressor_list.extend([list(pair) for pair in itertools.combinations(opts,2)])  
  
ls = ['Faster_Payment_Inbound_Retail','Faster_Payment_Outbound_Retail','Retail Debit Card ATM Us On Us']

tuning_results = pd.DataFrame(columns=['combination','mape'])
results_store = []
results_store.append(tuning_results)

for idx,regr in enumerate(regressor_list):
  for idx2,item in enumerate(ls):
    df = pd.read_csv('../data/'+item+'.csv')
    df = df[['reporting_date','volume']]
    df = df.sort_values(by='reporting_date')
    df.columns = ['ds','y']
    
    if regr != 'baseline':
      df = regressors.produce_flags(df[['ds','y']])
      df = df[['ds','y']+regr]
    
    m = Prophet()

    if regr != 'baseline':
      for col in regr:
        m.add_regressor(col)

    m.fit(df)

    initial = str(len(df)-(30*7)) + 'days'

    df_cv = cross_validation(m,initial=initial, period = '30 days', horizon='30 days', parallel="processes")

    for i in df_cv['cutoff'].unique():
      df_p = performance_metrics(df_cv[(df_cv['cutoff']==i)], rolling_window=1)
      results_store.append(pd.DataFrame([[idx,(df_p['smape'].values[0]*100)]],columns=['combination','mape']))

all_results = pd.concat(results_store).reset_index()[['combination','mape']]
all_results['combination'] = all_results['combination'].replace('business_start','bm_start')
all_results['combination'] = all_results['combination'].replace('business_end','bm_end')

def plotthing():
  fig,ax = plt.subplots()
  
  sns.boxplot(
      data=all_results, x="combination", y="mape",
      flierprops={"marker": "x"},
      boxprops={"linewidth": 1}
  )
  
  # Set the ytick locations and labels, can also use np array here
  ax.set_ylabel("MAPE (%)")
  #ax.set_xticklabels(['baseline','holiday','friday','monday','weekend','bm_start','bm_end'],fontsize=10)
  
  # Show the plot
  plt.show()

plotthing()
