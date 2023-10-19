import pandas as pd
import datetime
from datetime import date, timedelta
from dateutil.relativedelta import *

bankholidaylist = ['2017-01-02', '2017-04-14','2017-04-17','2017-05-01', '2017-05-29',
                  '2017-08-28','2017-12-25', '2017-12-26','2018-01-01','2018-03-30', 
                  '2018-04-02','2018-05-07','2018-05-28', '2018-08-27','2018-12-25',
                  '2018-12-26','2019-01-01','2019-04-19', '2019-04-22','2019-05-06',
                  '2019-05-27', '2019-08-26','2019-12-25','2019-12-26','2020-01-01',
                  '2020-04-10', '2020-04-13','2020-05-08','2020-05-25', '2020-08-31',
                  '2020-12-25','2020-12-26','2021-01-01',
                  '2021-04-02', '2021-04-05','2021-05-03','2021-05-31', '2021-08-30',
                  '2021-12-28','2021-12-27','2022-01-03','2022-01-04','2022-04-15'
                  ,'2022-04-18','2022-05-02','2022-06-02','2022-06-03','2022-08-29',
                  '2022-09-19', '2022-12-26', '2022-12-27', '2023-01-02', '2023-04-07',
                  '2023-04-10', '2023-05-01', '2023-05-08', '2023-05-29', '2023-08-28',
                  '2023-12-25', '2023-12-26','2024-01-01','2024-03-29','2024-04-01','2024-05-06','2024-05-27','2024-08-26','2024-12-25','2024-12-26',
                  '2025-01-01','2025-04-18','2025-04-21','2025-05-05','2025-05-26','2025-08-25','2025-12-25','2025-12-26']
  
def produce_flags(df):
  df['holiday'] = df['ds'].apply(lambda x: 1 if str(x)[:10] in bankholidaylist else 0)

  df['monday'] = df['ds'].apply(lambda x: 1 if datetime.datetime.strptime(str(x)[:10], "%Y-%m-%d").weekday() == 0 else 0)
  df['tuesday'] = df['ds'].apply(lambda x: 1 if datetime.datetime.strptime(str(x)[:10], "%Y-%m-%d").weekday() == 1 else 0)
  df['wednesday'] = df['ds'].apply(lambda x: 1 if datetime.datetime.strptime(str(x)[:10], "%Y-%m-%d").weekday() == 2 else 0)
  df['thursday'] = df['ds'].apply(lambda x: 1 if datetime.datetime.strptime(str(x)[:10], "%Y-%m-%d").weekday() == 3 else 0)
  df['friday'] = df['ds'].apply(lambda x: 1 if datetime.datetime.strptime(str(x)[:10], "%Y-%m-%d").weekday() == 4 else 0)
  df['weekend'] = df['ds'].apply(lambda x: 1 if datetime.datetime.strptime(str(x)[:10], "%Y-%m-%d").weekday() > 4 else 0)

  df['first3_weekday'] = df['ds'].apply(lambda x: 1 if datetime.datetime.strptime(str(x)[:10], "%Y-%m-%d").weekday() in [0,1,2] else 0)

  bm_start = pd.date_range(df['ds'].min(), df['ds'].max(), freq='BMS')
  bm_end = pd.date_range(df['ds'].min(), df['ds'].max(),freq='BM')
  df['business_start'] = df['ds'].apply(lambda x: (x in bm_start))
  df['business_start'] = df['business_start'].astype(int)
  df['business_end'] = df['ds'].apply(lambda x: (x in bm_end))
  df['business_end'] = df['business_end'].astype(int)
  
  bm_end_lag = pd.to_datetime([x - timedelta(days=1) for x in bm_end])
  df['business_end_lag'] = df['ds'].apply(lambda x: (x in bm_end_lag))
  df['business_end_lag'] = df['business_end_lag'].astype(int)
  
  bm_end_lf=[]
  for dt in bm_end:
    if dt.weekday!=4:
      bm_end_lf.append(dt+relativedelta(weekday=FR(-1)))
    else:
      bm_end_lf.append(dt)
  bm_end_lf_lag = pd.to_datetime([x - timedelta(days=1) for x in bm_end_lf])
  bm_end_lf = pd.to_datetime([x for x in bm_end_lf])
  
  df['business_end_lf'] = df['ds'].apply(lambda x: (x in bm_end_lf))
  df['business_end_lf'] = df['business_end_lf'].astype(int)
  df['business_end_lf_lag'] = df['ds'].apply(lambda x: (x in bm_end_lf_lag))
  df['business_end_lf_lag'] = df['business_end_lf_lag'].astype(int)
  
  return df