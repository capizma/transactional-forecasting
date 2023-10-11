import pandas as pd

def ingest_dataframe(fname):
    
    df = pd.read_csv('./data/'+fname)
    df = df[['reporting_date','volume']]
    df = df.sort_values(by='reporting_date')
    df.columns = ['ds','y']
    return df