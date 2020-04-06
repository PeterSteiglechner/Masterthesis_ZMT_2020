#!/usr/bin/env python

# PETER STEIGLECHNER, 10.3.2020

# Take config file, and save that

import config
import xarray as xr 
import numpy as np
import pandas as pd 



def final_saving():
    ''' Saves the aggregated Numbers to a ncdf file'''
    times = np.arange(0,config.N_timesteps)
    ds = xr.Dataset(
        {
            "total_agents":("time", config.nr_agents),
            "total_population": ("time", config.nr_pop),
            "total_deaths": ("time", config.nr_deaths),
            "total_trees": ("time", config.nr_trees)#
            #"agents":(("ag_ind", "properties", "time"), t1)
        },
        {"time": times}#, "ag_ind": indices_agents, "properties": properties},
    )

    ds.attrs['folder']=config.folder

    ds.attrs['map_type']=config.map_type
    ds.attrs['agent_type']=config.agent_type
    for key in config.params.keys():
        ds.attrs["Param: "+key]=config.params[key]
    
    ds.attrs['init_option']=config.init_option
    ds.attrs['Init N_agents'] = config.N_agents
    ds.attrs['Init N_trees']=config.N_trees 

    if config.map_type=="EI":
        for key in config.TreeDensityConditionParams.keys():
            ds.attrs["Tree Cond: "+key]=config.TreeDensityConditionParams[key]
    
    ds.attrs['N_timesteps']=config.N_timesteps
    ds.attrs['updatewithreplacement'] = int(config.updatewithreplacement)
    
    ds.to_netcdf(path=config.folder+"Statistics.ncdf")
    return 

def produce_agents_DataFrame(t):
    if len(config.agents)>0:
        all_reports = np.array([ag.report()[0] for ag in config.agents])
        properties = config.agents[0].report()[1]
        df = pd.DataFrame(
            {
                key: all_reports[:,n] for n, key in enumerate(properties[:])
            }, 
            columns=properties[:]
        )
        df = df.set_index(properties[0])
        #df.attrs["Time"]=t
        df.to_csv(config.folder+"AgentsStats_t="+str(t)+".csv")

        data = config.EI.tree_density
        midpoints = config.EI.EI_midpoints
        df_env = pd.DataFrame(
            {
                'Midpoints_x': midpoints[:,0],
                'Midpoints_y': midpoints[:,1],
                'TreeDensity': data
            }, 
            #columns=['Midpoints_x', 'Midpoints_y','TreeDensity']
        )
        df_env = df_env.set_index(['Midpoints_x', 'Midpoints_y'])
        #df.attrs["Time"]=t
        df_env.to_csv(config.folder+"AgentsStats_t="+str(t)+"_env.csv")
    
    # TODO! Environment as DataSet!
    return 

'''
ds = xr.Dataset(
    {
        "total_population": ("time", nr_pop),
        "total_agents":("time", nr_agents),
        "agents":(("ag_ind", "properties", "time"), t1)
    },
    {"time": times, "ag_ind": indices_agents, "properties": properties},
)
'''