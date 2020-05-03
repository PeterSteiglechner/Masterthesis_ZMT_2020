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
import xarray as xr


################################
####  Load Model functions  ####
################################
#from CreateMap_class import Map   #EasterIslands/Code/Triangulation/")
from CreateEI_ExtWaterExtAgricPaper import Map
from observefunc import observe, plot_TreeMap_only, plot_agricultureSites_onTop, plot_agents_on_top, plot_statistics
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
config.N_agents = 3
config.N_trees = 12e6
config.firstSettlers_init_pop=int(25)
config.childrenPop=int(15)
config.firstSettlers_moving_raidus = 0.6
config.init_TreePreference = 0.8
config.index_count=0


### RUN
config.updatewithreplacement = False
config.StartTime = 800
config.EndTime=1900
config.N_timesteps = config.EndTime-config.StartTime
config.seed= int(sys.argv[1])
print("Seed: ", config.seed)
#config.folder= LATER

### HOUSEHOLD PARAMS
config.MaxSettlementSlope=9
#config.MaxSettlementElev=300
#config.MinSettlementElev=10
config.Penalty50_SettlementElev = 100

config.max_pop_per_household_mean = 4*12  # Bahn2017 2-3 houses with each a dozen people. But I assume they also should include children
config.max_pop_per_household_std = 5



config.LowerLimit_PopInHousehold = 7

config.MaxPopulationDensity=173*2  #500 # DIamond says 90-450 ppl/mile^2  i.e. 34 to 173ppl/km^2 BUT THIS WILL BE HIGHER IN CENTRES OF COURSE
#config.MaxAgricPenalty = 100

config.tree_need_per_capita = 5 # Brandt Merico 2015 h_t =5 roughly.
config.MinTreeNeed=0.3      #Fraction
config.TreePref_kappa = 5


config.BestTreeNr_forNewSpot = 20*(config.max_pop_per_household_mean+config.max_pop_per_household_std)*config.tree_need_per_capita #HowManyTreesAnAgentWantsInARadiusWhenItMoves = 



#config.Nr_AgricStages = 10
#config.dStage = ((1-config.MinTreeNeed)/config.Nr_AgricStages)

#config.agricSites_need_per_Capita = 2./6.  # Flenley Bahn
HighorLowFix= sys.argv[2]
if HighorLowFix=="highFix":
    config.agricYield_need_per_Capita = 0.5 # Puleston High N  in ACRE!
else:
    config.agricYield_need_per_Capita = 1.7


config.maxNeededAgric = np.ceil((config.max_pop_per_household_mean+config.max_pop_per_household_std)*config.agricYield_need_per_Capita)


#YearsOfDecreasingTreePref = 1400-config.StartTime #400
#config.treePref_decrease_per_year = (config.init_TreePreference - config.MinTreeNeed)/YearsOfDecreasingTreePref #0.002
#config.treePref_change_per_BadYear = 0.1 # UPDATE: NOT FRACTION ANYMORE, but actual percentage of tree pref! #1. # THIS IS THE FRACTION of a dStage! #0.02 

#config.HowManyDieInPopulationShock = 10
#NOT NEEDED ANYMORE: config.FractionDeathsInPopShock = 0.2

# Each agent can (in the future) have these parameters different
config.params={'tree_search_radius': 2, 
        'agriculture_radius':1,
        'moving_radius': 15,
        'moving_radius_later':5,
        'reproduction_rate': 0.007,  # logistically Max from BrandtMerico2015 # 0.007 from Bahn Flenley
        }
config.PopulationMovingRestriction = 5000

config.MaxFisherAgents = 10


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
config.tree_pop_timespan  = 10

config.UpperLandSoilQuality=0.1
config.ErodedSoilYield=0.75
#config.YearsBeforeErosionDegradation=20
# need to make sure that 80% (RULL) are covered
#config.AgriConds={'minElev':20,'maxElev_highQu':250,
#    'maxSlope_highQu':3.5,'maxElev_lowQu':380,'maxSlope_lowQu':6,
#    'MaxWaterPenalty':300,}
config.gridpoints_y=75
config.AngleThreshold = 30
# config.m2_to_acre = FIXED
# config.km2_to_acre = FIXED

config.MapPenalty_Kappa = 5

config.tree_regrowth_rate=0.05 # every 20 years all trees are back?  # Brandt Merico 2015# Brander Taylor 0.04 % Logisitc Growth rate.


# DROUGHTS
config.drought_RanoRaraku_1=[config.StartTime, 1100]

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
config.tree_growth_poss = np.where(config.EI.carrying_cap>0)[0].astype(int)

# Plot/Test the resulting Map.
#print("Test Map: ",config.EI.get_triangle_of_point([1e4, 1e4], config.EI_triObject))

config.EI.get_Map_Penalty()

print("################   DONE: MAP ###############")


#################################################
########   Folder and Run Prepa   ###############
#################################################

withTreegrowth = sys.argv[3]  # "withRegrowth"

# TEST TREE REGRWOTH
#config.EI.tree_density = 0.5*config.EI.tree_density
#config.EI.tree_density-= (np.random.random(config.EI.N_els)* (config.EI.tree_density).astype(float)).astype(int)
#config.EI.tree_density[np.random.choice(np.arange(0,config.EI.N_els), size=1000, replace =False).astype(int)] = 0
#config.EI.agriculture[np.random.choice(np.arange(0,config.EI.N_els), size=300, replace =False).astype(int)] = 1
config.folder = "Figs_WithDrought/"
#config.folder+="TreeRegrowthTest"
config.folder += "FullModel_grid"+str(config.gridpoints_y)+"_repr"+'%.0e' % (config.params['reproduction_rate'])+"_mv"+"%.0f" % config.params['moving_radius']+"_"+withTreegrowth+"_"+HighorLowFix+"_seed"+str(config.seed)+"/"
#"FullModel_"+config.agent_type+"_"+config.map_type+"_"+config.init_option+"/"
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
    config.EI.check_drought(0, config.drought_RanoRaraku_1, config.EI_triObject)
    init_agents()
    #observe(config.StartTime, fig=None, ax=None, specific_ag_to_follow=config.agents[0], save=True)
    observe(config.StartTime, fig=None, ax=None, specific_ag_to_follow=None, save=True)
    plt.close()

    #observe(config.StartTime)
    for t in np.arange(config.StartTime,config.EndTime+1):#, num=config.N_timesteps):

        update_time_step(t)  
        if withTreegrowth=="withRegrowth": 
            regrow_update(config.tree_regrowth_rate)
        else:
            config.TreeRegrowth.append(0)
        agric_update()
        if withTreegrowth=="withRegrowth": 
            popuptrees(t)
        else:
            config.TreePopUp.append(0)

        config.EI.check_drought(t, config.drought_RanoRaraku_1, config.EI_triObject)
        #observe(t+1)
        
        if not np.isnan(config.timeSwitchedMovingRad) and len(config.agents)>0 and config.nr_pop[-1] >= config.PopulationMovingRestriction :
            for ag in config.agents:
                ag.moving_radius=config.params["moving_radius_later"]
            config.timeSwitchedMovingRad=t
            print("Agents reduced their Moving Radius")

        #observe(t+1, fig=None, ax=None, specific_ag_to_follow=None, save=True)
        if len(config.agents)>0:
            if (t+1)%50==0:
                if len(config.agents)>0:
                    ag = config.agents[0]
                else:
                    ag = None
                observe(t+1, fig=None, ax=None, specific_ag_to_follow=ag, save=True)
                plt.close()
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

    data = xr.open_dataset(config.folder+"Statistics.ncdf")
    plot_statistics(data, config.folder)

