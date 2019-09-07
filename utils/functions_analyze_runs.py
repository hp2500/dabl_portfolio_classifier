import openml
import numpy as np
import pandas as pd
import json
import re



# function that counts runs and flows per task
def count_runs(task_id, evaluation_metric = 'area_under_roc_curve'):
    
    # get list of evaluations for each task
    evals = openml.evaluations.list_evaluations(function=evaluation_metric,
                                                task = [task_id],
                                                output_format='dataframe')
    
    # only take the ones that refer to sklearn runs
    evals_runs = evals[evals['flow_name'].str.contains('sklearn')]
    
    # drop duplicate flows
    evals_flows = evals_runs.drop_duplicates(subset = ['flow_id'])
    
    return(len(evals_runs), len(evals_flows))



# function to extract best runs from task 
def task_to_runs(task_id, 
                 evaluation_metric = 'area_under_roc_curve', 
                 cutoff_best = 5, 
                 keep_duplicates = True):

    # get list of evaluations for each task
    evals = openml.evaluations.list_evaluations(function=evaluation_metric, 
                                            task = [task_id],
                                            output_format='dataframe')

    # only take the ones that refer to sklearn runs
    evals = evals[evals['flow_name'].str.contains('sklearn')]
    
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
    return params



# function to save classifier information for list of tasks to dict
def get_best_classifiers(tasks, 
                         evaluation_metric = 'area_under_roc_curve', 
                         cutoff_best = 5, 
                         keep_duplicates = True):
   
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
                                 keep_duplicates=keep_duplicates)

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
        nr_runs, nr_flows = count_runs(task_id)
        
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

    