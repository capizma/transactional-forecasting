from exploration.utils import forecasting
from exploration.utils import dataframe_utils, regressors, cutoff_utils, json_utils

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

logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

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
    
    for fname in os.listdir('./data/'):
        
        regressor_groups = json_utils.read_file_config(fname)[1]
    
        # Skip if file already output / dataset too small
        if json_utils.read_params(fname, ['hyperfit']) != 0:
            continue
    
        df = dataframe_utils.ingest_dataframe(fname)
        
        if len(df) < 100:
            print(fname + " has too few rows. Skipping...")
            continue
        
        best_params = None
        best_smape = 1
        
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
                
            if np.min(smapes) < best_smape:
                best_params = grid[np.argmin(smapes)]
                best_smape = np.min(smapes)
                
        if best_params == None:
            problematic_fits.append(fname)
        
        json_utils.append_data(fname,"hyperfit",best_params)
        json_utils.append_data(fname,"initial_smape",best_smape)
        
        if len(problematic_fits) > 0:
            print("WARNING - Some data could not be fit appropriately.")
            print("Please inspect the data quality for the following files:")
            for f in problematic_fits:
                print(f)

    