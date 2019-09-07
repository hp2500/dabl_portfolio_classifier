import openml
import numpy as np
import json
import pandas as pd


# define function for submodular optimization
def sub_opt(df, n):
    
    # rank setups according to mean performace over tasks
    ranks = df.mean(axis = 1).rank(method = 'first', ascending = False)
    
    # get best setup as starting point
    best_set = list(df[ranks == 1].index)
    
    # loop to sequentially identify setups that add most value to portfolio
    for i in range(n):
    
        # get the best evals of any classifier in set for each task
        best_evals = df.loc[best_set].max()
        # print(best_evals.values)
        # print()
        
        # differences between best set evals and other evals
        diff = df - best_evals.values
        
        # drop negative values (cases in which the portfolio would perform better)
        diff_pos = diff[diff > 0]
        
        # replaces nas by zero so that means will be based on the same denominator
        diff_pos = diff_pos.fillna(0)
        
        # get setup that maximizes aggregate performance 
        # (largest average positive difference to ensemble performance)
        next_rank = diff_pos.mean(axis = 1).rank(method = 'first', ascending = False)
        
        # append setup id to set of best setups
        best_set.append(next_rank[next_rank == 1].index[0])
    
    return(best_set)


# computes performance improvement for each additional classifier
def perf_imp(df):
    
    # rank setups according to mean performace over tasks
    ranks = df.mean(axis = 1).rank(method = 'first', ascending = False)
    
    # get best setup as starting point
    best_set = list(df[ranks == 1].index)
    performance = []
    
    # loop to sequentially identify setups that add most value to ensemble
    for i in range(len(df)):
    
        # get the best evals of any classifier in set for each task
        best_evals = df.loc[best_set].max()
            
        if best_evals.mean() == 1:
            performance.append(best_evals.mean())
            break
            
        performance.append(best_evals.mean())
        
        # differences between best set evals and other evals
        diff = df - best_evals.values
        
        # drop negative values (cases in which the portfolio would perform better)
        diff_pos = diff[diff > 0]
        
        # replaces nas by zero so that means will be based on the same denominator
        diff_pos = diff_pos.fillna(0)
        
        # get setup that maximizes aggregate performance 
        # (largest average positive difference to ensemble performance)
        next_rank = diff_pos.mean(axis = 1).rank(method = 'first', ascending = False)
        
        # append setup id to set of best setups
        best_set.append(next_rank[next_rank == 1].index[0])
                
        setup = openml.setups.get_setup(next_rank[next_rank == 1].index[0])
        flow = openml.flows.get_flow(setup.flow_id)
        last_step = flow.components[json.loads(flow.parameters['steps'])[-1]['value']['step_name']]
        print(last_step.name)
    
    return(best_set, performance)




