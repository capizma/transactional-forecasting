from exploration.utils import dataframe_utils
from run_monthly_testing import produce_ticks

import sys
import pandas as pd
import numpy as np
import datetime
from datetime import date, datetime, timedelta
import os
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import tqdm
import logging
pd.options.mode.chained_assignment = None 

def plotthing(df):
    fig,ax = plt.subplots()
    sns.lineplot(data=df,x='ds',y='y')
    sns.lineplot(data=df,x='ds',y='yhat').set(title=fname)

    ax.fill_between(x=df.loc[df["y"] != None, "ds"],
                    y1=df.loc[df["y"] != None, "yhat_lower"],
                    y2=df.loc[df["y"] != None, "yhat_upper"], alpha=0.4)
    
    ticks = produce_ticks(df)
    plt.xticks(ticks[0],ticks[1])
    #ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    
    plt.savefig('./results/monthly/predict_images/'+fname.replace('.csv','.png'))
    
    return ax

for fname in tqdm.tqdm(os.listdir('./data/')):
    
    df = dataframe_utils.ingest_dataframe(fname)
    
    if len(df) > 100:
        print(fname + " has been identified as daily. Skipping...")
        continue
    
    m = Prophet()
    
    m.fit(df)
    
    future = m.make_future_dataframe(3,freq='M')
    future.iloc[-3:] = future.iloc[-3:] + timedelta(days=1)
    
    forecast = m.predict(future)
    forecast['ds'] = forecast['ds'].apply(lambda x : str(x)[:10])
    
    forecast['yhat'],forecast['yhat_lower'],forecast['yhat_upper'] = forecast['yhat'].astype(int), forecast['yhat_lower'].astype(int), forecast['yhat_upper'].astype(int)
    
    df['yhat'],df['yhat_lower'],df['yhat_upper'] = np.nan,np.nan,np.nan
    forecast = pd.concat([df,forecast.iloc[len(df):]])
    
    plotthing(forecast.iloc[len(forecast)-180:])
    
    forecast = forecast[['ds','yhat','y']]
    forecast['y'] = forecast['y'].fillna(0)
    forecast['value_type'] = forecast['y'].apply(lambda x: "PREDICTED" if x == 0 else "ACTUAL")
    
    forecast['yearmonth'] = forecast['ds'].apply(lambda x: str(x)[:7])
    
    forecast['y'] = forecast.apply(lambda x: x['yhat'] if x['y'] == 0 else x['y'],axis=1)
    forecast['volume'] = forecast['y']
    forecast = forecast[['yearmonth','volume','value_type']]
    
    forecast.to_csv('./results/monthly/monthly_results/'+fname)
    
print('\nAll results exported to ./results/monthly/monthly_results/')