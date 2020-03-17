
# coding: utf-8

# # Best runs on all tasks

# ## Import modules

# In[1]:


# import modules
import openml
from openml import tasks, flows, runs
import sklearn
from sklearn import feature_selection
from sklearn.svm import SVC
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
from dabl import detect_types

# set api key
openml.config.apikey = open('.key', 'r').readline().strip('\n')

def get_run_info_svc(run_id):
    
    run = openml.runs.get_run(run_id)
    flow = openml.flows.get_flow(run.flow_id)

    if "estimator" in flow.parameters:
        flow = openml.flows.get_flow(flow.components['estimator'].flow_id)
        
    last_step = flow.components[json.loads(flow.parameters['steps'])[-1]['value']['step_name']]
    setup = openml.setups.get_setup(run.setup_id)
    last_step_parameters = [v for v in setup.parameters.values() if v.flow_id == last_step.flow_id]
    params = {p.parameter_name: p.value for p in last_step_parameters}
    
    param_keys = ('C', 'coef0', 'degree', 'gamma', 'kernel', 'max_iter')
    params = dict((k, params[k]) for k in param_keys)
        
    params['C'] = eval(params['C'].replace('"',''))
    params['coef0'] = eval(params['coef0'].replace('"',''))
    params['degree'] = eval(params['degree'].replace('"',''))
    params['gamma'] = eval(params['gamma'].replace('"',''))
    params['kernel'] = params['kernel'].replace('"','')
    params['max_iter'] = eval(params['max_iter'].replace('"',''))
    params['random_state'] = 1
    params['probability'] = True
    return params
# In[2]:


# get all supervised classification tasks
tasks_all = openml.tasks.list_tasks(task_type_id=1, output_format='dataframe', tag = 'OpenML-CC18')
# drop problematic tasks
tasks_all = tasks_all.drop([3573, 146825, 167121, 167124])


# ## Get OpenML runs for SVC flows

# In[3]:


# get SVC evals
good_flows = [6246, 6952, 8330, 6954, 7756, 5499, 8317, 7223, 6009, 7707, 6269, 5983, 16374, 16347, 16345]
evals = openml.evaluations.list_evaluations('area_under_roc_curve',
                                            flow= good_flows, 
                                            task=list(tasks_all.tid),
                                            output_format='dataframe'
                                            )

# rank evaluations
evals['rank'] = evals.groupby('task_id')['value'].rank('first', ascending=False)

# get best evaluations
best_evals = evals.loc[evals['rank'] <= 5]


# In[4]:


# drop problematic runs
best_evals = best_evals[best_evals.run_id != 6148258]
best_evals = best_evals[best_evals.run_id != 8231647]


# In[5]:


best_evals.shape


# ## Check categorical / numerical / mixed features

# In[6]:


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


# In[24]:


cat_num


# In[8]:


# check distribution
cat_num['cat_num'].value_counts()


# In[9]:


# check ids of mixed feature tasks
list(cat_num.tid.loc[cat_num.cat_num == 'mixed'])


# In[20]:


task_ids = cat_num[cat_num.cat_num != 'mixed'].tid


# ## Loop over all tasks

# In[25]:


# infinite loop
while 1:

    # randomly sample a task
    # i = tasks_all.tid.sample().iloc[0] # this samples from all tasks
    i = task_ids.sample() # sample from numeric or categorical only
    
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
        
        print('Run', k, 'on task', i)
        print(datetime.now())
        
        try:
            # get params
            params = get_run_info_svc(k)

            # define classifier
            clf = SVC(**params)

            # pick pipeline according to feature types
            if not any(categorical_indicator):
                pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), clf)
            elif all(categorical_indicator):
                pipe = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'), clf)
            else:
                print('Skip task with mixed features...')
                break
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

