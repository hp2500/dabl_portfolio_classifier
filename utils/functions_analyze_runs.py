import openml
import numpy as np
import pandas as pd
import json
import re

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score



# function that counts runs and flows per task
def count_runs(task_id, evaluation_metric = 'area_under_roc_curve', cutoff_date = '2100-01-01 00:00:00'):
    
    # get list of evaluations for each task
    evals = openml.evaluations.list_evaluations(function=evaluation_metric,
                                                task = [task_id],
                                                output_format='dataframe')
    
    # only take the ones that refer to sklearn runs
    evals_runs = evals[evals['flow_name'].str.contains('sklearn')]

    # only take the runs that were published before cutoff date
    evals_runs = evals_runs.loc[evals_runs.upload_time <  cutoff_date]
    
    # drop duplicate flows
    evals_flows = evals_runs.drop_duplicates(subset = ['flow_id'])
    
    return(len(evals_runs), len(evals_flows))



# function to extract best runs from task 
def task_to_runs(task_id, 
                 evaluation_metric = 'area_under_roc_curve', 
                 cutoff_best = 5, 
                 keep_duplicates = True,
                 cutoff_date = '2100-01-01 00:00:00'):

    # get list of evaluations for each task
    evals = openml.evaluations.list_evaluations(function=evaluation_metric, 
                                            task = [task_id],
                                            output_format='dataframe')

    # only take the ones that refer to sklearn runs
    evals = evals[evals['flow_name'].str.contains('sklearn')]

    # only take the runs that were published before cutoff date
    evals = evals.loc[evals.upload_time <  cutoff_date]
    
    # drop duplicate flows 
    if keep_duplicates == False:
        evals = evals.drop_duplicates(subset = ['flow_id'])

    # take the best performing ones for the given task according to cutoff
    if cutoff_best < 1.0:
        best_runs = evals.nlargest(int(len(evals)*cutoff_best), 'value')
    else:
        best_runs = evals.nlargest(cutoff_best, 'value')
    
    return(best_runs)


# function extracts classifier component from run 
def str_to_clf(s): 
    
    s = s.split('base_estimator')[0]
    s = s.split('preprocessor')[0]
    s = s.rpartition('.')[-1]
    s = re.sub(r'[^\w\s]','',s)
    s = re.sub(r'[0-9]+', '', s)
                        
    return(s)



def get_run_info(run_id):
    
    run = openml.runs.get_run(run_id)
    flow = openml.flows.get_flow(run.flow_id)
    
    if "estimator" in flow.parameters:
        flow = openml.flows.get_flow(flow.components['estimator'].flow_id)
        last_step = flow.components[json.loads(flow.parameters['steps'])[-1]['value']['step_name']]
        setup = openml.setups.get_setup(run.setup_id)
        last_step_parameters = [v for v in setup.parameters.values() if v.flow_id == last_step.flow_id]
        parameters = {p.parameter_name: p.value for p in last_step_parameters}
        name = str_to_clf(last_step.name)
        return name, parameters
    
    if len(flow.components) == 0:
        name = str_to_clf(flow.class_name)
        parameters = dict(flow.parameters)
        return name, parameters
        
    last_step = flow.components[json.loads(flow.parameters['steps'])[-1]['value']['step_name']]
    setup = openml.setups.get_setup(run.setup_id)
    last_step_parameters = [v for v in setup.parameters.values() if v.flow_id == last_step.flow_id]
    parameters = {p.parameter_name: p.value for p in last_step_parameters}
    name = str_to_clf(last_step.name)

    return name, parameters


# function to get model params
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


# function to get model params
def get_run_info_lr(run_id):
    
    run = openml.runs.get_run(run_id)
    flow = openml.flows.get_flow(run.flow_id)

    if "estimator" in flow.parameters:
        flow = openml.flows.get_flow(flow.components['estimator'].flow_id)
        
    last_step = flow.components[json.loads(flow.parameters['steps'])[-1]['value']['step_name']]
    setup = openml.setups.get_setup(run.setup_id)
    last_step_parameters = [v for v in setup.parameters.values() if v.flow_id == last_step.flow_id]
    params = {p.parameter_name: p.value for p in last_step_parameters}
    
    param_keys = ('C', 'dual', 'fit_intercept', 'intercept_scaling', 'max_iter', 'solver', 
                  'penalty', 'multi_class', 'l1_ratio')
    params = dict((k, params[k]) for k in param_keys)
    
    params['C'] = eval(params['C'].replace('"',''))
    params['dual'] = eval(params['dual'].replace('"','').capitalize())
    params['fit_intercept'] = eval(params['fit_intercept'].replace('"','').capitalize())
    params['intercept_scaling'] = eval(params['intercept_scaling'].replace('"',''))
    params['max_iter'] = eval(params['max_iter'].replace('"',''))
    params['solver'] = params['solver'].replace('"','')
    
    params['penalty'] = params['penalty'].replace('"','')
    params['multi_class'] = params['multi_class'].replace('"','')
    
    if params['l1_ratio'].replace('"','') != 'null':
        params['l1_ratio'] = eval(params['l1_ratio'].replace('"',''))
    else: 
        params['l1_ratio'] = None

    params['random_state'] = 1
    return params



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
        params['min_impurity_decrease'] = 0
    try:
        params['min_impurity_split'] = eval(params['min_impurity_split'].replace('"',''))
    except:
        params['min_impurity_split'] = 0
        
    params['min_impurity_decrease'] = params.pop('min_impurity_split')

    
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


# function to get model params
def get_run_info_svc_str(run_id):
    
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
        
    params['C'] = params['C'].replace('"','')
    params['coef0'] = params['coef0'].replace('"','')
    params['degree'] = params['degree'].replace('"','')
    params['gamma'] = params['gamma'].replace('"','')
    params['kernel'] = params['kernel'].replace('"','')
    params['max_iter'] = params['max_iter'].replace('"','')
    params['random_state'] = 1
    params['probability'] = True
    return params



# function to save classifier information for list of tasks to dict
def get_best_classifiers(tasks, 
                         evaluation_metric = 'area_under_roc_curve', 
                         cutoff_best = 5, 
                         keep_duplicates = True,
                         cutoff_date = '2100-01-01 00:00:00'):
   
    # create empty dict to populate with tasks
    dict_tasks = {}

    # loop over all task ids 
    for task_id in tasks['tid']:

        # print task id
        print('Task ID:', task_id)

        # get best runs for each task
        best_runs = task_to_runs(task_id, 
                                 evaluation_metric= evaluation_metric,
                                 cutoff_best=cutoff_best,
                                 keep_duplicates=keep_duplicates,
                                 cutoff_date = cutoff_date)

        # create empry dict to populate with clf
        dict_runs = {}

        # loop over best runs for each task
        for run_id in best_runs['run_id']:

            # print flow id 
            print(run_id, '', end = '')
            
            # get clf name and parameters from run
            try: 
                dict_clf, dict_params = get_run_info(run_id)

                clf_props = {'clf_name': dict_clf,
                            'clf_params': dict_params}

                # add to dict_flow
                dict_runs[run_id] = clf_props
                
            except Exception as e:
                print(e, end = '')
            
        print()
        
        # count number of runs and flows for each task
        nr_runs, nr_flows = count_runs(task_id, cutoff_date = cutoff_date)
        
        # add to dict of runs per task
        run_all_data = {'nr_runs': nr_runs,
                        'nr_flows': nr_flows,
                        'run_data': dict_runs} 

        # add to overall dict of tasks
        dict_tasks[task_id] = run_all_data
        
    # return dict
    return(dict_tasks)




# function to save classifier information for list of tasks to dict
def count_classifiers(tasks, keep_duplicates = True):
    
    all_runs = pd.DataFrame()

    # loop over all task ids 
    for task_id in tasks['tid']:

        # print task id
        print('Task ID:', task_id)

        # get best runs for each task
        runs = task_to_runs(task_id, 
                             evaluation_metric = 'area_under_roc_curve',
                             cutoff_best= 1000000,
                             keep_duplicates=keep_duplicates)

        runs['clf_name'] = runs.flow_name.apply(str_to_clf)
        
        all_runs = all_runs.append(runs)
        
    # return dict
    return(all_runs)

def svc_param_test(task_id, run_id, params):

    # get task
    i = task_id
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

    k = run_id

    print('Run', k, 'on task', i)
    print(datetime.now())

    try:

        # define classifier
        clf = SVC(**params)

        # pick pipeline according to feature types
        if not any(categorical_indicator):
            pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), clf)
        elif all(categorical_indicator):
            pipe = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'), clf)
        else:
            pipe = make_pipeline(SimpleImputer(strategy='most_frequent'), preprocessor, clf)

        # run best model on the task
        run = openml.runs.run_model_on_task(pipe, task, avoid_duplicate_runs=False)

        # print feedbackack
        print('Publish openml run...')

        # push tag
        #run.push_tag('best_models')

        # publish the run 
        run.publish()
        # print feedback
        print('View run online: https://www.openml.org/r/' + str(run.run_id))
        print('Setup', openml.runs.get_run(run.run_id).setup_id)
        print('Flow', openml.runs.get_run(run.run_id).flow_id)
        print()

    except Exception as e:
        print(e)
        
    return run


def svc_param_test_local(task_id, params, cv = 2):

    # get task
    i = task_id
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

    try:

        # define classifier
        clf = SVC(**params)
        scoring = make_scorer(roc_auc_score, needs_proba=True)

        # pick pipeline according to feature types
        if not any(categorical_indicator):
            pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), clf)
        elif all(categorical_indicator):
            pipe = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'), clf)
        else:
            pipe = make_pipeline(SimpleImputer(strategy='most_frequent'), preprocessor, clf)

        run_cv = cross_val_score(pipe, X = X, y = y, scoring=scoring, cv = cv)

        return run_cv

    except Exception as e:
        print(e)
        
    


    


    
