import pandas as pd
import numpy as np
import datetime
from datetime import date, datetime, timedelta
import os
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from exploration.utils import regressors
from prophet.diagnostics import cross_validation, performance_metrics
import tqdm
import logging
pd.options.mode.chained_assignment = None 

def produce_single_cutoff(df):
    ls = []
    
    df['yearmonth'] = df['ds'].apply(lambda x: str(x)[:7])
    
    subset = df[(df['yearmonth'] == df['yearmonth'].unique()[-3])]
    ls.append(subset['ds'].values[0])
        
    n_days = (datetime.strptime(str(df['ds'].max())[:10],'%Y-%m-%d') - datetime.strptime(subset['ds'].values[0],'%Y-%m-%d')).days
    
    if n_days < 80:
        subset = df[(df['yearmonth'] == df['yearmonth'].unique()[-4])]
        ls = []
        
        ls.append(subset['ds'].values[0])
            
        n_days = (datetime.strptime(str(df['ds'].max())[:10],'%Y-%m-%d') - datetime.strptime(subset['ds'].values[0],'%Y-%m-%d')).days
    
    return [pd.to_datetime(ls),str(n_days) + ' days']

def run_crossvalidation(df,m):
    # CV going back 90 periods, this might not be 90 days
    
    print("Running CV for "+str(df.iloc[-90]['ds'])+" - "+str(df['ds'].max()))
        
    temp = datetime.strptime(df['ds'].max(),"%Y-%m-%d")
    temp_past = datetime.strptime(df.iloc[-90]['ds'],"%Y-%m-%d")
    n_days = str((date(temp.year,temp.month,temp.day) - date(temp_past.year,temp_past.month,temp_past.day)).days) + " days"
    
    df_cv = cross_validation(m,cutoffs=pd.to_datetime([temp_past]), horizon=n_days, parallel="processes")
    
    df_p = performance_metrics(df_cv, rolling_window=1)
    
    return [df_p['smape'].values[0],str(df.iloc[-90]['ds'])+" - "+str(df['ds'].max())]
    
def run_forecast(fname,future_date):
    
    df = pd.read_csv('./data/'+fname)
    df = df[['reporting_date','volume']]
    df = df.sort_values(by='reporting_date')
    df.columns = ['ds','y']
    
    temp = datetime.strptime(df['ds'].max(),"%Y-%m-%d")
    n_days = (future_date - date(temp.year,temp.month,temp.day)).days
    
    regr = eval(pd.read_csv('./params/hyperfits/'+fname)['regressor'].values[0])
    params = eval(pd.read_csv('./params//hyperparams/'+fname)['params'].values[0])
    
    m = Prophet(**params)
    
    df = regressors.produce_flags(df)
    df = df[['ds','y']+regr]
    
    for i in df.columns[2:]:
        m.add_regressor(i)
    
    m.add_country_holidays(country_name='UK')
    
    m.fit(df)  # Fit model with given params
    
    cv = run_crossvalidation(df, m)
    
    future = m.make_future_dataframe(periods=n_days)
    
    future = regressors.produce_flags(future)
    future = future[['ds']+regr]
    
    forecast = m.predict(future)
    
    forecast['ds'] = forecast['ds'].apply(lambda x: datetime.strftime(x,'%Y-%m-%d'))
    
    forecast['y'] = np.nan
    
    forecast = forecast.iloc[len(df):]
    forecast = forecast[['ds','y','yhat','yhat_lower','yhat_upper']]
    df = df[['ds','y']]
    
    forecast = pd.concat([df,forecast])
    
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
    
    return [forecast,cv[0],cv[1]]
    
def run_past_forecast(fname,start_date,end_date):
    
    df = pd.read_csv('./data/'+fname)
    df = df[['reporting_date','volume']]
    df = df.sort_values(by='reporting_date')
    df.columns = ['ds','y']
    
    all_data = df.copy()
    
    df = df[(df['ds'] < datetime.strftime(start_date,'%Y-%m-%d'))]
    
    temp = datetime.strptime(df['ds'].max(),"%Y-%m-%d")
    n_days = (end_date - start_date).days + 1
    
    regr = eval(pd.read_csv('./params/hyperfits/'+fname)['regressor'].values[0])
    params = eval(pd.read_csv('./params//hyperparams/'+fname)['params'].values[0])
    
    m = Prophet(**params)
    
    df = regressors.produce_flags(df)
    df = df[['ds','y']+regr]
    
    for i in df.columns[2:]:
        m.add_regressor(i)
    
    m.add_country_holidays(country_name='UK')
    
    print("Running past forecast for "+str(n_days)+ " days with cutoff period after "+str(df['ds'].max()))
    
    m.fit(df)  # Fit model with given params
    
    cv = run_crossvalidation(df, m)
    
    future = m.make_future_dataframe(periods=n_days)
    
    future = regressors.produce_flags(future)
    future = future[['ds']+regr]
    
    forecast = m.predict(future)
    
    forecast['ds'] = forecast['ds'].apply(lambda x: datetime.strftime(x,'%Y-%m-%d'))
    
    forecast['y'] = np.nan
    
    forecast = forecast.iloc[len(df):]
    forecast = forecast[['ds','yhat','yhat_lower','yhat_upper']]
    
    forecast = all_data.merge(forecast,how='left',on=['ds'])
    
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
    
    return [forecast,cv[0],cv[1]]
    