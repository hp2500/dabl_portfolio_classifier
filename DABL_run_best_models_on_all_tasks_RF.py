# import modules

import openml
from openml import tasks, flows, runs
import sklearn
from sklearn import feature_selection
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pprint
from collections import OrderedDict, Counter
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
import re
import random
import numpy as np
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import json
from itertools import combinations
import signal


# set api key
openml.config.apikey = open('.key', 'r').readline().strip('\n')



# function to get model params
def get_run_info_rf(run_id):
    
    run = openml.runs.get_run(run_id)
    flow = openml.flows.get_flow(run.flow_id)

    if "estimator" in flow.parameters:
        flow = openml.flows.get_flow(flow.components['estimator'].flow_id)
        
    last_step = flow.components[json.loads(flow.parameters['steps'])[-1]['value']['step_name']]
    setup = openml.setups.get_setup(run.setup_id)
    last_step_parameters = [v for v in setup.parameters.values() if v.flow_id == last_step.flow_id]
    params = {p.parameter_name: p.value for p in last_step_parameters}
    
    param_keys = ('n_estimators', 'criterion', 'max_depth', 'min_samples_split', 
                  'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features',
                  'max_leaf_nodes', 'min_impurity_split',
                  'bootstrap', 'oob_score', 'n_jobs', 'random_state', 'class_weight')
    
    # couldnt use min_impurity_decrease, ccp_alpha, max_samples, ccp_alpha
    # all of the following could be done in a loop... implement when I have time
    
    params = dict((k, params[k]) for k in param_keys)
        
    params['n_estimators'] = eval(params['n_estimators'].replace('"',''))
    params['criterion'] = params['criterion'].replace('"','')   
    try:
        params['max_depth'] = eval(params['max_depth'].replace('"',''))
    except: 
        params['max_depth'] = None
    params['min_samples_split'] = eval(params['min_samples_split'].replace('"',''))
    params['min_samples_leaf'] = eval(params['min_samples_leaf'].replace('"',''))
    params['min_weight_fraction_leaf'] = eval(params['min_weight_fraction_leaf'].replace('"',''))
    
    if params['max_features'].replace('"','') == 'auto':
       params['max_features'] = params['max_features'].replace('"','')
    else:
        params['max_features'] = eval(params['max_features'].replace('"',''))
    try:
        params['max_leaf_nodes'] = eval(params['max_leaf_nodes'].replace('"',''))
    except:
        params['max_leaf_nodes'] = None
    try:
        params['min_impurity_decrease'] = eval(params['min_impurity_decrease'].replace('"',''))
    except:
        params['min_impurity_decrease'] = 1e-7
    try:
        params['min_impurity_split'] = eval(params['min_impurity_split'].replace('"',''))
    except:
        params['min_impurity_split'] = 0
    
    params['bootstrap'] = eval(params['bootstrap'].replace('"','').capitalize())
    params['oob_score'] = eval(params['oob_score'].replace('"','').capitalize())
    try:
        params['n_jobs'] = eval(params['n_jobs'].replace('"',''))
    except:
        params['n_jobs'] = None
        
    params['random_state'] = 1

    try:
        if params['class_weight'].replace('"','') == 'None' or params['class_weight'].replace('"','') == "null":
            params['class_weight'] = None
        else:
            params['class_weight'] = params['class_weight'].replace('"','')
    except:
        print('No class_weight specified, choose default.')

    return params



# get all supervised classification tasks
tasks_all = openml.tasks.list_tasks(task_type_id=1, output_format='dataframe', tag = 'OpenML-CC18')
# drop problematic tasks
tasks_all = tasks_all.drop([3573, 146825, 167121, 167124])


# get SVC evals
good_flows = [5804, 8365, 5909, 8918, 6969, 8315, 8351]
evals = openml.evaluations.list_evaluations('area_under_roc_curve',
                                            flow= good_flows, 
                                            task=list(tasks_all.tid),
                                            output_format='dataframe'
                                            )

# rank evaluations
evals['rank'] = evals.groupby('task_id')['value'].rank('first', ascending=False)

# get best evaluations
best_evals = evals.loc[evals['rank'] <= 5]

# empty list to populate with feature types
types = []

for i in tasks_all.tid:
    print(i, '', end = '')
    
    # get task
    task = openml.tasks.get_task(i)

    # get dataset object 
    data = openml.datasets.get_dataset(task.dataset_id)

    # get relevant info from dataset object
    X, y, categorical_indicator, attribute_names = data.get_data(dataset_format='array',
                                                                target=data.default_target_attribute)
    
    if not any(categorical_indicator):
        types.append((i, 'numeric'))
    elif all(categorical_indicator):
        types.append((i, 'categorical'))
    else:
        types.append((i, 'mixed'))

cat_num = pd.DataFrame(types, columns = ['tid', 'cat_num'])
cat_num = pd.DataFrame(types, columns=['tid', 'cat_num'])



# define timeout handler
def handler(signum, frame):
    raise Exception("Timeout!")
    
# Register the signal function handler
signal.signal(signal.SIGALRM, handler)


# infinite loop
while 1:

    # randomly sample a task
    i = tasks_all.tid.sample().iloc[0] # this samples from all tasks
    #i = task_ids.sample() # sample from numeric or categorical only
    
    # get task
    task = openml.tasks.get_task(i)
    
    # get dataset object
    data = openml.datasets.get_dataset(task.dataset_id)

    # get relevant info from dataset object
    X, y, categorical_indicator, attribute_names = data.get_data(dataset_format='array',
                                                                target=data.default_target_attribute)

    # mask with feature types
    cat = categorical_indicator
    num = [not k for k in categorical_indicator]

    # create column transformers
    numeric_transformer = make_pipeline(#SimpleImputer(strategy='median'), 
                                        StandardScaler())

    categorical_transformer = make_pipeline(#SimpleImputer(strategy='most_frequent'),
                                            OneHotEncoder(handle_unknown='ignore'))

    preprocessor = ColumnTransformer(
    transformers=[
    ('num', numeric_transformer, num),
    ('cat', categorical_transformer, cat)])
    
    # loop over runs in random order
    for k in best_evals.run_id.sample(frac=1):
        
        # set time limit
        signal.alarm(3600)
        
        print('Run', k, 'on task', i)
        print(datetime.now())
        
        try:
            # get params
            params = get_run_info_rf(k)

            # define classifier
            clf = RandomForestClassifier(**params)

            # pick pipeline according to feature types
            if not any(categorical_indicator):
                pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), clf)
            elif all(categorical_indicator):
                pipe = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'), clf)
            else:
                pipe = make_pipeline(SimpleImputer(strategy='most_frequent'), preprocessor, clf)
                
            # run best model on the task
            run = openml.runs.run_model_on_task(pipe, task, avoid_duplicate_runs=True)

            # print feedbackack
            print('Publish openml run...')

            # push tag
            # run.push_tag('best_models')
            # publish the run
            
            run.publish()
            # print feedback
            print('View run online: https://www.openml.org/r/' + str(run.run_id))
            print('Setup', openml.runs.get_run(run.run_id).setup_id)
            print('Flow', openml.runs.get_run(run.run_id).flow_id)
            print()

        except Exception as e:
            print(e)
