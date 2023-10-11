import forecasting
from exploration.utils import dataframe_utils, regressors, cutoff_utils

import os
from prophet import Prophet
import pandas as pd
import datetime
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
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
import multiprocessing
sns.set(style="darkgrid")

warnings.simplefilter(action='ignore')

logging.getLogger("fbprophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

if __name__ == '__main__':

    problematic_fits = []
    
    param_grid = {
        'changepoint_prior_scale':[0.005,0.05,0.1],
        'seasonality_prior_scale':[0.05,0.5,10.0],
        'holidays_prior_scale':[0.05,0.5,10.0],
        'yearly_seasonality':[True,False],
    }

    for fname in os.listdir('./params/hyperfits/'):
    
        # Skip if file already output / dataset too small
        if fname in os.listdir('./params/hyperparams/'):
            continue
    
        df = dataframe_utils.ingest_dataframe(fname)
        
        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        smapes = []
        
        cutoffs = cutoff_utils.produce_general_cutoffs(df)      
        regr_df = pd.read_csv("./params/hyperfits/"+fname)
        selected_regr = eval(regr_df['regressor'].values[0])
    
        best_params = None
        best_smape = 0
        
        for param_set in tqdm.tqdm(all_params):
            try:
                smape = forecasting.produce_cv(df,params=param_set,regr=selected_regr,cutoffs=cutoffs)[0]
            
                smapes.append(smape)
            except:
                # Unable to forecast - move to next grid item
                smapes.append(0)
                pass
            
        if np.min(smapes) == 0:
            # We have a problematic fit
            continue
            
        if np.min(smapes) > best_smape:
            best_params = all_params[np.argmin(smapes)]
            best_smape = np.min(smapes)
                
        if best_params == None:
            problematic_fits.append(fname)
        
        export = pd.DataFrame([[best_params,best_smape]],columns=['params','smape'])
        export.to_csv('./params/hyperfits/'+fname)  
        problematic_fits.append(fname)
        
        print('\nAll regressors exported to ./params/hyperparams/')
        if len(problematic_fits) > 0:
            print("WARNING - Some data could not be fit appropriately.")
            print("Please inspect the data quality for the following files:")
            for f in problematic_fits:
                print(f)

    