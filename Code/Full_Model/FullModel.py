# 
# #!/usr/bin/env/ python

#Peter Steiglechner
# 18.3.2020
# 1.4.2020 Extension with Water and Agriculture!
# 6.4.2020 DONE! Moving works, Reprod works, Feeding works I think.

# This Model is the full model

################################
####  Configure Global Vars ####
################################
import config

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
from copy import copy
from time import time
import sys 
import os
from pathlib import Path   # for creating a new directory
import shutil  # for copying files into a folder

from mpl_toolkits.axes_grid1 import make_axes_locatable

import pickle



################################
####  Load Model functions  ####
################################
#from CreateMap_class import Map   #EasterIslands/Code/Triangulation/")
from CreateEI_ExtWaterExtAgricPaper import Map
from observefunc import observe, observe_density, plot_movingProb
from agents import agent, init_agents#, init_agents_EI
#from agents import Tree
from update import update_single_agent, update_time_step
from LogisticRegrowth import regrow_update, agric_update, popuptrees  # EasterIslands/Code/TreeRegrowth_spatExt/")

from write_to_ncdf import final_saving#,  produce_agents_DataFrame
#from analyse_ncdf import plot_statistics

################################
####     Specify Model      ####
################################
#config.map_type = "EI+Water"
#config.agent_type = "household"
#config.init_option = "anakena"

###  INIT
config.N_agents = 5
config.N_trees = 12e6
config.init_pop=int(15)
config.init_TreePreference = 0.8
config.index_count=0

### RUN
config.updatewithreplacement = False
config.StartTime = 800
config.EndTime=1850
config.N_timesteps = config.EndTime-config.StartTime
config.seed= 12
#config.folder= LATER

### HOUSEHOLD PARAMS
config.MaxSettlementSlope=6
#config.MaxSettlementElev=300
#config.MinSettlementElev=10
config.SweetPointSettlementElev = 0

config.max_pop_per_household_mean = 3.5*12  # Bahn2017 2-3 houses with each a dozen people. But I assume they also should include children
config.max_pop_per_household_std = 0.5*12 



LowerLimit_PopInHousehold = 10

config.MaxPopulationDensity=173*3  #500 # DIamond says 90-450 ppl/mile^2  i.e. 34 to 173ppl/km^2 BUT THIS WILL BE HIGHER IN CENTRES OF COURSE
#config.MaxAgricPenalty = 100

config.tree_need_per_capita = 5 # Brandt Merico 2015 h_t =5 roughly.
config.MinTreeNeed=0.3      #Fraction

config.BestTreeNr_forNewSpot = 10*(config.max_pop_per_household_mean+config.max_pop_per_household_std)*config.tree_need_per_capita #HowManyTreesAnAgentWantsInARadiusWhenItMoves = 



#config.Nr_AgricStages = 10
#config.dStage = ((1-config.MinTreeNeed)/config.Nr_AgricStages)

#config.agricSites_need_per_Capita = 2./6.  # Flenley Bahn
config.agricYield_need_per_Capita = 0.5  # Puleston High N  in ACRE!

config.maxNeededAgric = np.ceil((config.max_pop_per_household_mean+config.max_pop_per_household_std)*config.agricYield_need_per_Capita)


#YearsOfDecreasingTreePref = 1400-config.StartTime #400
#config.treePref_decrease_per_year = (config.init_TreePreference - config.MinTreeNeed)/YearsOfDecreasingTreePref #0.002
#config.treePref_change_per_BadYear = 0.1 # UPDATE: NOT FRACTION ANYMORE, but actual percentage of tree pref! #1. # THIS IS THE FRACTION of a dStage! #0.02 

#config.HowManyDieInPopulationShock = 10
#NOT NEEDED ANYMORE: config.FractionDeathsInPopShock = 0.2

# Each agent can (in the future) have these parameters different
config.params={'tree_search_radius': 1.6, 
        'agriculture_radius':0.8,
        'moving_radius': 8,
        'reproduction_rate': 0.007,  # logistically Max from BrandtMerico2015 # 0.007 from Bahn Flenley
        }

# DEFAULT RUN
#config.alpha_w= 0.3
#config.alpha_t = 0.2
#config.alpha_p = 0.0
#config.alpha_a = 0.3
#config.alpha_m = 0.2
config.alpha_w= 0.2
config.alpha_t = 0.2
config.alpha_p = 0.2
config.alpha_a = 0.2
config.alpha_m = 0.2

config.PenalToProb_Prefactor = 10

#NrOfEquivalentBestSitesToMove= 10


#### MAP

# LEVELS 1 or 2
#config.TreeDensityConditionParams = {'minElev':10, 'maxElev': 400,'maxSlope': 7}
config.TreeDensityConditionParams = {
    'minElev':5, 'maxSlopeHighD': 5, "ElevHighD":250, 
    'maxElev': 430,'maxSlope':8.5,
    "factorBetweenHighandLowTreeDensity":2}

config.tree_pop_percentage = 0.01
config.tree_pop_timespan  = 5

config.UpperLandSoilQuality=0.2
config.ErodedSoilYield=0.7
#config.YearsBeforeErosionDegradation=20
# need to make sure that 80% (RULL) are covered
#config.AgriConds={'minElev':20,'maxElev_highQu':250,
#    'maxSlope_highQu':3.5,'maxElev_lowQu':380,'maxSlope_lowQu':6,
#    'MaxWaterPenalty':300,}
config.gridpoints_y=100
config.AngleThreshold = 30
# config.m2_to_acre = FIXED
# config.km2_to_acre = FIXED

config.tree_regrowth_rate=0.05 # every 20 years all trees are back?  # Brandt Merico 2015# Brander Taylor 0.04 % Logisitc Growth rate.


##############################################
#####    Load or Create MAP     ##############
##############################################
print("################   LOAD / CREATE MAP ###############")
NewMap=True

#if NewMap==False and (not config.N_init_trees==12e6 or not config.):
#   print("Probably you should create a new Map!!")

filename = "Map/EI_grid"+str(config.gridpoints_y)+"_rad"+str(config.params['tree_search_radius'])+"+"+str(config.params['agriculture_radius'])
if Path(filename).is_file() and NewMap==False:
    with open(filename, "rb") as EIfile:
        config.EI = pickle.load(EIfile)
    if not config.EI.TreeNeighbourhoodRad == config.params['tree_search_radius'] or not config.EI.AgricNeighbourhoodRad==config.params['agriculture_radius']:
        print("The Map you loaded does not fit to the Params specified in FullModel.py (resource search radius)")
        quit()
else:
    config.EI = Map(config.gridpoints_y,N_init_trees = config.N_trees, angleThreshold=config.AngleThreshold)
    with open(filename, "wb") as EIfile:
        pickle.dump(config.EI, EIfile)
    config.EI.plot_agriculture(which="high", save=True)
    config.EI.plot_agriculture(which="low", save=True)
    config.EI.plot_agriculture(which="both", save=True)
    config.EI.plot_TreeMap_and_hist(name_append="T12e6", hist=True, save=True)
    config.EI.plot_water_penalty()

config.EI_triObject = mpl.tri.Triangulation(config.EI.points_EI_km[:,0], config.EI.points_EI_km[:,1], triangles = config.EI.all_triangles, mask=config.EI.mask)

# Plot/Test the resulting Map.
#print("Test Map: ",config.EI.get_triangle_of_point([1e4, 1e4], config.EI_triObject))

print("################   DONE: MAP ###############")


#################################################
########   Folder and Run Prepa   ###############
#################################################

config.folder = "Figs/"
config.folder += "FullModel_grid"+str(config.gridpoints_y)+"_repr"+'%.0e' % (config.params['reproduction_rate'])+"_mv"+"%.0f" % config.params['moving_radius']+"_Standard/"#"FullModel_"+config.agent_type+"_"+config.map_type+"_"+config.init_option+"/"
config.analysisOn=True
#string = [item+r":   "+str(vars(config)[item]) for item in dir(config) if not item.startswith("__")]
#for bla in string: print(bla)


################################
####     Run   Model        ####
################################
np.random.seed(config.seed)
def run():
    ''' run the model specified in the config file'''
    # Save the current state of the implementation to the Subfolder
    Path(config.folder).mkdir(parents=True, exist_ok=True)
    Path(config.folder+"/USED_programs/").mkdir(parents=True, exist_ok=True)
    for file in ["FullModel.py","config.py", "agents.py", "CreateEI_ExtWaterExtAgricPaper.py", 
                "LogisticRegrowth.py", "observefunc.py", "update.py"]:
        shutil.copy(file, config.folder+"/USED_programs/")
    print("Working in ",config.folder,"; Python Scripts have been copied into subfolder USED_programs")
    
    print("################   RUN ###############")
    config.Array_populationOccupancy = np.array([[] for _ in range(config.EI.N_els)]).reshape((config.EI.N_els,0))
    config.Array_tree_density = np.array([[] for _ in range(config.EI.N_els)]).reshape((config.EI.N_els,0))
    config.Array_agriculture = np.array([[] for _ in range(config.EI.N_els)]).reshape((config.EI.N_els,0))
    # RUN
    init_agents()
    #observe(config.StartTime)
    for t in np.arange(config.StartTime,config.EndTime+1):#, num=config.N_timesteps):


        update_time_step(t)  
        regrow_update(config.tree_regrowth_rate)
        agric_update()
        popuptrees(t)
        #observe(t+1)

        if len(config.agents)>0:

            ag = config.agents[0]
            print("Following agent ",ag.index, "(pop",ag.pop,"): Pref ", '%.4f' % ag.treePref,", TreeNeed ", ag.tree_need, ", AgriSite/Need ", len(ag.AgricSites), "/",ag.AgriNeed, " happy:",ag.happy)
        
            if (t+1)%10==0:
                observe(t+1)
            if (t+1)%50==0:
                plt.cla()
                _,ax = config.EI.plot_agricultureSites(ax=None, fig=None, save=False, CrossesOrFacecolor="Facecolor")
                ax.set_title("Time "+str(t+1))
                plt.savefig(config.folder+"AgricDensityMap_t"+str(t+1)+".png")
                plt.close()
        
    print("Finished running Model in folder: ", config.folder)


        

if __name__ == "__main__":
    run()
    #plot_dynamics(config.N_timesteps)
    final_saving()
    #plot_statistics(config.folder)


