import pandas as pd
import json
import os

with open('./params/params.json','r') as f:
    params_file = json.loads(f.read())
    
def read_file_config(fname):
    with open('./params/groups/config.json','r') as f:
        groups = json.loads(f.read())
    with open('./params/groups/datafiles.json','r') as f:
        datafile = json.loads(f.read())
    try:
        params = groups[datafile[fname]['param_group']+'_param_group']
    except:
        params = groups['standard_param_group']
        
    try:
        regressors = groups[datafile[fname]['param_group']+'_regressor_group']
    except:
        regressors = groups['standard_regressor_group']
        
    return [params,regressors]

def check_file_exists(fname,filepath):
    with open('./params/groups/datafiles.json','r') as f:
        datafile = json.loads(f.read())
    try:
        with open(filepath,'r') as f:
            datafile = json.loads(f.read())
            value = datafile[fname]
            return True
    except:
        return False
    
def read_params(fname,params):
    with open('./params/params.json','r') as f:
        params_file = json.loads(f.read())
    if 'hyperparams' in params:
        try:
            params = params_file[fname]['hyperparams']
            return params
        except:
            pass
        
    if 'hyperfit' in params:
        try:
            regressors = params_file[fname]['hyperfit']
            return regressors
        except:
            pass
        
    return 0

def append_data(fname,param_type,params):
    with open('./params/params.json','r') as f:
        params_file = json.loads(f.read())
    try:
        params_file[fname][param_type] = params
            
        with open('./params/params.json','w') as f:
            json.dump(params_file,f,indent=4)
            
        return 1
    except:
        params_file[fname] = {}
        
        # Avoiding stackoverflow
        with open('./params/params.json','w') as f:
            json.dump(params_file,f,indent=4)
        
        append_data(fname,param_type,params)
        
