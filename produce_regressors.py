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

def regressor_grid(regressor_ls):
    # Build further regression options
    ls = list(itertools.product(range(len(regressor_ls)-1),repeat = len(regressor_ls)))
    map_func = lambda x,y : regressor_ls[x] if y != 0 else None
    unique_regressors = []
    
    for item in ls:
        mapped = [map_func(x,y) for x,y in enumerate(item)]
        mapped = set(mapped)
        try:
            mapped.remove(None)
        except KeyError:
            pass
        unique_regressors.append(list(mapped))
    
    unique_regressors = unique_regressors[1:]
    
    return unique_regressors

if __name__ == '__main__':

    problematic_fits = []
    
    regressor_groups = [
        ['business_end_lf_lag','business_end_lag','business_end','business_end_lf'],
        ['business_start','business_end','holiday']  
    ]

    for fname in os.listdir('./data/'):
    
        # Skip if file already output / dataset too small
        if fname in os.listdir('./params/hyperfits/'):
            continue
    
        df = dataframe_utils.ingest_dataframe(fname)
        
        if len(df) < 100:
            print(fname + " has been identified as monthly. Skipping...")
            continue
        
        best_params = None
        best_smape = 0
        
        for group in regressor_groups:
            
            grid = regressor_grid(group)
            smapes = []
            cutoffs = cutoff_utils.produce_general_cutoffs(df)
            
            for regr in tqdm.tqdm(grid):
                try:
                    smape = forecasting.produce_cv(df,params=None,regr=regr,cutoffs=cutoffs)[0]
                
                    smapes.append(smape)
                except:
                    # Unable to forecast - move to next grid item
                    smapes.append(0)
                    pass
                
            if np.min(smapes) == 0:
                # We have a problematic fit
                continue
                
            if np.min(smapes) > best_smape:
                best_params = grid[np.argmin(smapes)]
                best_smape = np.min(smapes)
                
        if best_params == None:
            problematic_fits.append(fname)
        
        export = pd.DataFrame([[best_params,best_smape]],columns=['regressor','smape'])
        export.to_csv('./params/hyperfits/'+fname)  
        problematic_fits.append(fname)
        
        print('\nAll regressors exported to ./params/hyperfits/')
        if len(problematic_fits) > 0:
            print("WARNING - Some data could not be fit appropriately.")
            print("Please inspect the data quality for the following files:")
            for f in problematic_fits:
                print(f)

    