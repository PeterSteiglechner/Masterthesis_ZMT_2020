#!/usr/bin/env/ python

#Peter Steiglechner
# 7.3.2020

# This Model has a class Model, where one can chooose between 
#   agents (indiviuals or households), using the spatial grid (default), Tree regrowth or not
#   Trees as agents or through density

################################
####  Configure Global Vars ####
################################
import config


import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from time import time
import sys 
from pathlib import Path   # for creating a new directory
import shutil  # for copying files into a folder




################################
####  Load Model functions  ####
################################
from CreateMap_class import Map   #EasterIslands/Code/Triangulation/")
from observefunc import observe, observe_density, observe_classical, plot_dynamics
from agents import agent, init_agents, init_agents_EI
from agents import init_agents_UnitSquare, Tree, init_trees
from update import update_single_agent, update_time_step
from LogisticRegrowth import regrow_update   # EasterIslands/Code/TreeRegrowth_spatExt/")

from write_to_ncdf import final_saving,  produce_agents_DataFrame
from analyse_ncdf import plot_statistics

################################
####     Specify Model      ####
################################
config.map_type = "EI"
config.agent_type = "household"
config.init_option = "anakena"
config.N_agents = 10
config.N_trees = 12e6
config.tree_regrowth_rate = 0.05 # every 20 years all trees are back?

config.updatewithreplacement = True
config.N_timesteps = 300

if config.map_type=="EI":
    config.gridpoints_y=70
    config.TreeDensityConditionParams = {'minElev':10, 'maxElev': 400,'maxSlope': 7}
    config.EI = Map(config.gridpoints_y,N_init_trees = config.N_trees)

'''
config.params_individual={'resource_search_radius': 0.1, 
        'moving_radius': 0.3,
        'fertility':0.05
        }
'''
config.params_hh={'resource_search_radius': 600, 
        'moving_radius': 3e3,
        'reproduction_rate': 0.09,
        'init_pop': 15,
        'tree_need_per_capita': 4,
        'max_pop_per_household': 40
        }

config.folder = "Figs/"
config.folder += "ThirstyVegs_a_"+config.agent_type+"_"+config.map_type+"_"+config.init_option+"/"

string = [item+r":   "+str(vars(config)[item]) for item in dir(config) if not item.startswith("__")]
for bla in string: print(bla)


################################
####     Run   Model        ####
################################
np.random.seed(100)
def run():
    ''' run the model specified in the config file'''
    # Save the current state of the implementation to the Subfolder
    Path(config.folder).mkdir(parents=True, exist_ok=True)
    Path(config.folder+"/USED_programs/").mkdir(parents=True, exist_ok=True)
    for file in ["HopVes.py","config.py", "agents.py", "CreateMap_class.py", 
                "LogisticRegrowth.py", "observefunc.py", "update.py"]:
        shutil.copy(file, config.folder+"/USED_programs/")
    print("Working in ",config.folder,"; Python Scripts have been copied into subfolder USED_programs")

    # RUN
    init_agents()
    if config.map_type=="unitsquare":
        init_trees()
    observe(0)
    for t in range(config.N_timesteps):
        update_time_step()  
        regrow_update(config.tree_regrowth_rate)
        if (t+1)%10==0:
            observe(t+1)
            produce_agents_DataFrame(t+1)

if __name__ == "__main__":
    run()
    #plot_dynamics(config.N_timesteps)
    final_saving()
    plot_statistics(config.folder)

