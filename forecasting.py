from exploration.utils import regressors, suppress_log

import os
from prophet import Prophet
import pandas as pd
import datetime
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import itertools
import logging
import json
import warnings
import tqdm
import numpy as np
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
import seaborn as sns
import multiprocessing

sns.set(style="darkgrid")

warnings.simplefilter(action='ignore')

logging.getLogger("fbprophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

def produce_cv(df,params=None,regr=None,cutoffs=None,horizon=None):
    with suppress_log.suppress_stdout_stderr():
        if params is not None:
            m = Prophet(**params) # Fit model with given params
        else:
            m = Prophet()
    
        m.add_country_holidays(country_name='UK')
    
        df = regressors.produce_flags(df[['ds','y']])
        
        if regr is not None:
            df = df[['ds','y']+regr]
    
            for col in df.columns[2:]:
                m.add_regressor(col)
    
        m.fit(df)
    
        if horizon is None:
            df_cv = cross_validation(m,cutoffs=cutoffs, horizon = '30 days', parallel="processes")
        else:
            df_cv = cross_validation(m,cutoffs=cutoffs, horizon = horizon, parallel="processes")
    
        df_p = performance_metrics(df_cv, rolling_window=1)

    return [df_p['smape'].values[0],df_cv]
