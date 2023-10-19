from exploration.utils import regressors, dataframe_utils, json_utils
from run_daily_testing import produce_ticks

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
import warnings
pd.options.mode.chained_assignment = None 

warnings.simplefilter(action='ignore')

def plotthing(df):
    fig,ax = plt.subplots()
    sns.lineplot(data=df,x='ds',y='y')
    sns.lineplot(data=df,x='ds',y='yhat').set(title=fname)

    ax.fill_between(x=df.loc[df["y"] != None, "ds"],
                    y1=df.loc[df["y"] != None, "yhat_lower"],
                    y2=df.loc[df["y"] != None, "yhat_upper"], alpha=0.4)
    
    ticks = produce_ticks(df)
    plt.xticks(ticks[0],ticks[1])
    
    plt.savefig('./results/predict_images/'+fname.replace('.csv','.png'))
    
    return ax

for fname in tqdm.tqdm(os.listdir('./data/')):
    
    # If there aren't any parameters then abandon the iteration
    if json_utils.read_params(fname, ['hyperfit']) == 0 or json_utils.read_params(fname, ['hyperparams']) == 0:
        print(fname + " doesn't have a full parameter list. Skipping...")
        continue
    
    exclude_holidays = False
    exclude_weekends = False
    
    df = dataframe_utils.ingest_dataframe(fname)
    
    # Checking for null values on holidays/weekends
    date_check = df.copy()
    date_check = regressors.produce_flags(date_check)
    date_range = pd.date_range(start=date_check['ds'].min(),end=date_check['ds'].max())
    temp = pd.DataFrame(pd.Series([datetime.strftime(x,'%Y-%m-%d') for x in date_range ],name='ds'))
    date_check = temp.merge(date_check,how='left',on=['ds'])
    
    # Holidays not in training data, take them out of predictions
    if len(date_check[((date_check['holiday'] > 0)) & (date_check['y'].isnull() == False)]) < 10:
        exclude_holidays = True
        
    # Weekends not in training data, take them out of predictions
    if len(date_check[((date_check['weekend'] > 0)) & (date_check['y'].isnull() == False)]) < 10:
        exclude_weekends = True
    
    # Calculating n_days to forecast
    end_date = datetime.strptime(df['ds'].max(),'%Y-%m-%d').date()
    date_range = pd.date_range(end_date,end_date + timedelta(days=365),freq='M')
    temp = datetime.strptime(df['ds'].max(),'%Y-%m-%d')
    
    # Accounting for spillover
    df['yearmonth'] = df['ds'].apply(lambda x: str(x)[:7].replace("-","")).astype(int)
    if len(df[(df['yearmonth'] == df['yearmonth'].unique()[-1])]) < 3:
        n_days = (date_range[2].date() - date(temp.year,temp.month,temp.day)).days
        df = df[(df['yearmonth'] < df['yearmonth'].unique()[-1])]
    else:
        n_days = (date_range[3].date() - date(temp.year,temp.month,temp.day)).days
    df = df.drop(['yearmonth'],axis=1)
    
    regr = json_utils.read_params(fname, ['hyperfit'])
    params = json_utils.read_params(fname, ['hyperparams'])
    
    m = Prophet(**params)
    
    df = regressors.produce_flags(df)
    df = df[['ds','y']+regr]
    
    for i in df.columns[2:]:
        m.add_regressor(i)
        
    m.add_country_holidays(country_name='UK')
    
    m.fit(df)
    
    future = m.make_future_dataframe(periods=n_days)
    
    future = regressors.produce_flags(future)
    
    if 'holiday' not in regr:
        regr.append('holiday')

    if 'weekend' not in regr:
        regr.append('weekend')

    future = future[['ds']+regr]
    
    if exclude_holidays == True:
        future = future[(future['holiday'] == 0)].reset_index().drop(['index'],axis=1)
    
    if exclude_weekends == True:
        future = future[(future['weekend'] == 0)].reset_index().drop(['index'],axis=1)
    
    forecast = m.predict(future[1:])
    forecast['ds'] = forecast['ds'].apply(lambda x: datetime.strftime(x,'%Y-%m-%d'))
    forecast['y'] = np.nan
    forecast = forecast.iloc[len(df):]
    forecast = forecast[['ds','y','yhat','yhat_lower','yhat_upper']]
    df = df[['ds','y']]
    
    forecast = pd.concat([df,forecast])
    
    forecast['yhat'],forecast['yhat_lower'],forecast['yhat_upper'] = forecast['yhat'].clip(lower=0),forecast['yhat_lower'].clip(lower=0),forecast['yhat_upper'].clip(lower=0)
    
    daily = forecast.copy()
    
    forecast['yearmonth'] = forecast['ds'].apply(lambda x: str(x)[:7])
    forecast = forecast[['y','yhat','yearmonth']]
    
    forecast = forecast.groupby(by=['yearmonth']).sum().reset_index()
    forecast['yhat'] = forecast['yhat'].astype(int)
    forecast['value_type'] = forecast['yhat'].apply(lambda x: "ACTUAL" if x == 0 else "PREDICTED")
    forecast['y'] = forecast.apply(lambda x: x['y'] if x['yhat'] == 0 else x['yhat'],axis=1)
    forecast['volume'] = forecast['y']
    forecast = forecast[['yearmonth','volume','value_type']]
    
    forecast.to_csv('./results/monthly_results_agg/'+fname)
    
    daily.to_csv('./results/daily_results/'+fname)
    
    plotthing(daily.iloc[-240:])
    
print('\nAll results exported to ./results/daily_results and ./results/monthly_results_agg/')
    
