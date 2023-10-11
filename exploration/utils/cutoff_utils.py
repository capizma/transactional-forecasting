import pandas as pd
import datetime
from datetime import date, timedelta, datetime

def produce_general_cutoffs(df):
  ls = []
  
  df['yearmonth'] = df['ds'].apply(lambda x: str(x)[:7])
  
  for i in range(7,1,-1):
    subset = df[(df['yearmonth'] == df['yearmonth'].unique()[-i])]
    ls.append(subset['ds'].values[0])
    
  return pd.to_datetime(ls)

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