#!/usr/bin/env/ python

# PETER STEIGLECHNER, 10.03.2020

# This file plots the previously generated ncdf file in `write_to_ncdf.py`


import xarray as xr 
import matplotlib.pyplot as plt
from observefunc import plot_dynamics
import pandas as pd 
import config

def plot_statistics(folder):
    #folder= "Figs/Test/HopVes_i_household_unitsquare_corner/"
    data = xr.open_dataset(folder+"Statistics.ncdf")
    print(data)

    ncdf_info = {'stats': [data.total_agents, data.total_population, data.total_deaths, data.total_trees] ,
            'agent_type': data.attrs['agent_type'], 
            'folder':folder }
    plot_dynamics(data.attrs['N_timesteps'], ncdf_info = ncdf_info)


#def plot_gif(folder):
#    data = xr.open_dataset(folder+"Statistics.ncdf")
#    config.agent_type = data.attrs['agent_type']
#    config.map_type = data.attrs['map_type']
#    config.init_option = data.attrs['init_option']
#    config.folder = data.attrs['folder']
#    config.N_timesteps = data.attrs['N_timesteps']
#    for t in range(config.N_timesteps):
#        ags = pd.read_csv(config.folder+"AgentsStats_t="+str(t)+".csv")
#        agents
