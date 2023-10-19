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

if __name__ == '__main__':

    problematic_fits = []

    for fname in os.listdir('./data/'):
        
        # If there aren't any regressors then abandon the iteration
        if json_utils.read_params(fname, ['hyperfit']) == 0:
            print(fname + " doesn't have any identified regressors. Skipping...")
            continue
        
        param_grid = json_utils.read_file_config(fname)[0]
    
        # Skip if file already output / dataset too small
        if json_utils.read_params(fname, ['hyperparams']) != 0:
            continue
    
        df = dataframe_utils.ingest_dataframe(fname)
        
        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        smapes = []
        
        cutoffs = cutoff_utils.produce_general_cutoffs(df)      
        selected_regr = json_utils.read_params(fname,"hyperfit")
    
        best_params = None
        best_smape = 1
        
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
            
        if np.min(smapes) < best_smape:
            best_params = all_params[np.argmin(smapes)]
            best_smape = np.min(smapes)
                
        if best_params == None:
            problematic_fits.append(fname)
        
        json_utils.append_data(fname,"hyperparams",best_params)
        json_utils.append_data(fname,"secondary_smape",best_smape)
        
        if len(problematic_fits) > 0:
            print("WARNING - Some data could not be fit appropriately.")
            print("Please inspect the data quality for the following files:")
            for f in problematic_fits:
                print(f)

    