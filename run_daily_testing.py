from exploration.utils import regressors, dataframe_utils, cutoff_utils
import forecasting

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

logging.getLogger("cmdstanpy").disabled = True 
logging.getLogger("prophet").disabled = True 
logging.getLogger("pandas").disabled = True 

sns.set_theme(style='darkgrid')

def produce_ticks(df):
    ls = []
    ls2 = []
    
    df['yearmonth'] = df['ds'].apply(lambda x: str(x)[:7]) 
    
    for idx,i in enumerate(df['yearmonth'].unique()):
        subset = df[(df['yearmonth'] == i)]
        
        if idx == 0 or idx == len(df['yearmonth'].unique())-1:
            ls.append(subset['ds'].values[0])
            ls2.append(datetime.strftime(datetime.strptime(subset['ds'].values[0],'%Y-%m-%d'),"%b-%y"))
        
    return [ls,ls2]

def plotthing(df):
    fig,ax = plt.subplots()
    sns.lineplot(data=df,x='ds',y='y')
    sns.lineplot(data=df,x='ds',y='yhat').set(title=fname + " - sMAPE "+str(round(results[0],2)))

    ax.fill_between(x=df.loc[df["y"] != None, "ds"],
                    y1=df.loc[df["y"] != None, "yhat_lower"],
                    y2=df.loc[df["y"] != None, "yhat_upper"], alpha=0.4)
    
    ticks = produce_ticks(df)
    plt.xticks(ticks[0],ticks[1])
    #ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    
    plt.savefig('./results/fit_images/'+fname.replace('.csv','.png'))
    
    return ax

if __name__ == '__main__':
    ls = []
    
    for fname in os.listdir('./params/hyperparams/'):
        data = pd.read_csv('./params/hyperparams/'+fname)
        data['fname'] = fname
        
        ls.append(data)
        
    smapes_ls = []
        
    data = pd.concat(ls).reset_index()
    data = data[['fname','smape']]
    data = data.sort_values(by='smape',ascending=False).reset_index()[['fname','smape']]
    
    for fname in tqdm.tqdm(os.listdir('./params/hyperparams/')):
        
        df = dataframe_utils.ingest_dataframe(fname)
        
        regr = eval(pd.read_csv('./params/hyperfits/'+fname)['regressor'].values[0])
        params = eval(pd.read_csv('./params/hyperparams/'+fname)['params'].values[0])
            
        cutoffs = cutoff_utils.produce_single_cutoff(df)
        
        results = forecasting.produce_cv(df,params=None,regr=regr,cutoffs=cutoffs[0],horizon=cutoffs[1])
        
        smapes_ls.append(pd.DataFrame([[fname,str(round(results[0],2))]],columns=['fname','smape']))
        
        forecast = results[1]
        forecast['ds'] = forecast['ds'].apply(lambda x: datetime.strftime(x,'%Y-%m-%d'))   
        forecast = forecast[['ds','y','yhat','yhat_lower','yhat_upper']]
        
        df['yhat_lower'],df['yhat_upper'],df['yhat'] = np.nan, np.nan, np.nan
        df = df[['ds','y','yhat','yhat_lower','yhat_upper']]
        
        forecast = pd.concat([df,forecast]).reset_index()
        
        plotthing(forecast.iloc[len(forecast)-240:])
        
    smapes_df = pd.concat(smapes_ls)
    smapes_df.to_csv('./results/cv_results.csv')
    
    print("\nAll 3-month range error metrics exported to ./results/cv_results.csv")   
    print("Results with sMAPE above 15% threshold (these are worth being inspected within /results/fit_images):")
    print(data[data['smape'] > 0.15])