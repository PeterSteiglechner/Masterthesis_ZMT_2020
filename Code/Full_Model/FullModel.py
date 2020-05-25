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
from copy import deepcopy
from time import time
import scipy.stats
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


seed = sys.argv[1] # number
HighorLowFix= sys.argv[2] # highFix, lowFix
TreeRegrowthOption = sys.argv[3]  # "withRegrowth", noRegrowth
TPrefOption = sys.argv[4]  # linear, delayed, logistic, careful
PopulationStability = sys.argv[5] # LessResPop, NormPop
AlphaSetting = sys.argv[6] # alphaStd, alphaHopping, alphaStd, alphaResource, alphaDeterministic



# Pre-settings
CreateNewMap = True
config.folder = "Figs_May21/"


################################
####     Init               ####
################################

config.index_count=0
config.N_agents_arrival = 2
config.Pop_arrival=int(40) # Good and Reuveny 2006, Brander Taylor 1998
config.r_M_arrival = 1  # first two agents move close to Anakena Beach

config.t_arrival = 800  # used in Bahn 2017?, Rull 2020 (800-1200)


################################
####     Map                ####
################################
config.gridpoints_y=50
config.AngleThreshold = 35

config.N_trees_arrival = 16e6  # Mieth and Bork 2015

TreePattern_evenly = {
    'minElev':0, 'maxSlopeHighD': 10, "ElevHighD":450, 
    'maxElev': 450,'maxSlope':10,
    "factorBetweenHighandLowTreeDensity":1}
TreePattern_twoLevel = {
    'minElev':0, 'maxSlopeHighD': 5, "ElevHighD":250, 
    'maxElev': 450,'maxSlope':10,
    "factorBetweenHighandLowTreeDensity":2}
config.TreeDensityConditionParams = TreePattern_evenly

config.F_PI_well=0.8 # Louwagie 2006
config.F_PI_poor=0.1 # Louwagie 2006
config.F_PI_eroded=0.5  # No Source 

# config.m2_to_acre = FIXED
# config.km2_to_acre = FIXED

if TreeRegrowthOption == "withRegrowth":
    config.tree_regrowth_rate= 0.05 # every 20 years all trees are back?  # Brandt Merico 2015# Brander Taylor 0.04 % Logisitc Growth rate.
    config.tree_pop_percentage = 0.005
    config.barrenYears_beforePopUp  = 10
elif TreeRegrowthOption=="noRegrowth":
    config.tree_regrowth_rate= 0.0 # every 20 years all trees are back?  # Brandt Merico 2015# Brander Taylor 0.04 % Logisitc Growth rate.
    config.tree_pop_percentage = 0
    config.barrenYears_beforePopUp  = 0
else:
    print("Error; wrong TreeGrowthOption")
    quit()
    
config.droughts_RanoRaraku=[[config.t_arrival, 1200], [1570, 1720]]






################################
####     Run                ####
################################config.updatewithreplacement = False
config.t_end=1900
config.seed= int(seed)



################################
####     Agents             ####
################################# 
# Section Harvest Environment interaction
config.r_T=2
config.r_F=1
config.T_Pref_max = 0.8    
config.T_Pref_min=0.2      #Fraction

if TPrefOption =="linear":
    config.f_T_Pref = lambda x: x*(config.T_Pref_max-config.T_Pref_min)+config.T_Pref_min
elif TPrefOption=="careful":
    config.f_T_Pref = lambda x: x**2*(config.T_Pref_max-config.T_Pref_min)+config.T_Pref_min
elif TPrefOption=="delayed":
    config.f_T_Pref  = lambda x: x**0.5*(config.T_Pref_max-config.T_Pref_min)+config.T_Pref_min
elif TPrefOption=="logistic":
    config.xi_T_Pref = 1/(config.T_Pref_max - config.T_Pref_min) * np.log(0.99/0.01)
    config.f_T_Pref = lambda x: config.logistic(x,config.xi_T_Pref, 0.5)*(config.T_Pref_max-config.T_Pref_min)+config.T_Pref_min
else:
    print("Error; Wront TPrefOption")
    quit()

config.T_Req_pP = 5 # Brandt Merico 2015 h_t =5 roughly.
if HighorLowFix=="highFix":
    config.F_Req_pP = 0.5 # Puleston High N  in ACRE!
elif HighorLowFix=="lowFix":
    config.F_Req_pP = 1.7
else:
    print("Error, wrong HighorLowFix Option")
#config.F_Req_pP = 2./6.  # Flenley Bahn

#config.FishingTabooYear = 1400
config.T_Pref_fisher_min=0.5
config.MaxFisherAgents=10

# Population dynamics
config.g_H1 = 1.007  # Bahn Flenley 2017 (with 800 A.D.)
config.SplitPop=int(12)  # 1 dozen = 1 dwelling 
config.pop_agent_max_mean = 3.5*12  # Bahn2017 2-3 houses with each a dozen people. 
config.pop_agent_max_std = 3
mu= config.pop_agent_max_mean-2*config.SplitPop
config.Split_k = mu**2/config.pop_agent_max_std**2
config.Split_theta = config.pop_agent_max_std**2 / mu

config.pop_agent_min = 6
if PopulationStability =="LessResPop":
    config.g_shape=3
    config.H_equ = 0.84422111
elif PopulationStability=="NormPop":
    config.g_shape =   1.95  # roughly tuned to get g(H_equ)=1
    config.H_equ = 0.68844221 # Puleston 2017
else:
    print("Error; wrong PopulationStability Option")
    quit()

# ALTERNATIVE: config.g_shape = 3
config.g_scale = 0.1 # as in Lee2008
config.g_scale_param = (config.g_H1)/ ( scipy.stats.gamma.cdf(1,config.g_shape, scale =config.g_scale)) # Lee 2008 and Puleston 2017

config.g_of_H = lambda x: config.g_scale_param* scipy.stats.gamma.cdf(x,config.g_shape, scale = config.g_scale) # Lee 2008 and Puleston 2017


# Moving 
config.r_M_init=100
config.r_M_later=5
config.pop_restrictedMoving = 5000
#config.k_X
#config.P_x 

# thresholds
config.w01 = 0.5**2/(np.pi*0.170**2)
config.w99= 5.0**2/(np.pi*0.170**2)
config.el01 = 0
config.el99 = 300
config.sl01=0
config.sl99=7.5
config.pd01=0
config.pd99=300 # Kirch 2010, Puleston 2017, rough estimate of local pop density in Hawaii and Maui
config.tr01=config.pd99*config.r_T**2/config.r_F**2*45*config.T_Pref_max*config.T_Req_pP
config.tr99=0 # to be determined ag.T_Pref_i* config.T_Req_pP* ag.pop * config.H_equ; 
config.trcondition = config.tr99 # to be determined
config.f01=(1-config.T_Pref_min)* config.F_Req_pP* 42
config.f99=0 # to be determined (1-ag.T_Pref_i)* config.F_Req_pP* ag.pop * config.H_equ; 
config.fcondition = config.f99 # to be determined

# Weights
#config.alpha_w= 0.3
#config.alpha_t = 0.2
#config.alpha_p = 0.0
#config.alpha_a = 0.3
#config.alpha_m = 0.2
if AlphaSetting=="alphaStd":
    config.alpha_W= 0.2
    config.alpha_T = 0.2
    config.alpha_D = 0.2
    config.alpha_F = 0.2
    config.alpha_G = 0.2

    # Penalty to Probability
    config.gamma = 20  
if not AlphaSetting=="alphaStd":
    if AlphaSetting=="alphaResource": 
        config.alpha_W= 0.0
        config.alpha_T = 0.5
        config.alpha_D = 0.0
        config.alpha_F = 0.5
        config.alpha_G = 0.0
        # Penalty to Probability
        config.gamma = 20  
    elif AlphaSetting=="alphaHopping":
        config.alpha_W= 0.0
        config.alpha_T = 0.2
        config.alpha_D = 0.0
        config.alpha_F = 0.2
        config.alpha_G = 0.0
        # Penalty to Probability
        config.gamma = 0.0
    elif AlphaSetting=="alphaDeterministic":
        config.alpha_W= 0.2
        config.alpha_T = 0.2
        config.alpha_D = 0.2
        config.alpha_F = 0.2
        config.alpha_G = 0.2
        # Penalty to Probability
        config.gamma = 1000 
    else:
        print("Error Alpha setting")
        quit()


################################
####     Agents             ####
################################# 

##############################################
#####    Load or Create MAP     ##############
##############################################
print("################   LOAD / CREATE MAP ###############")

filename = "Map/EI_grid"+str(config.gridpoints_y)+"_rad"+str(config.r_T)+"+"+str(config.r_F)+"+"+str(config.r_M_later)
if Path(filename).is_file() and CreateNewMap==False:
    with open(filename, "rb") as EIfile:
        config.EI = pickle.load(EIfile)
    if not config.EI.r_T == config.r_T or not config.EI.r_F==config.r_F or not config.EI.r_M_later==config.r_M_later:
        print("The Map loaded does not fit to the Params specified in FullModel.py (e.g. r_T)")
        quit()
else:
    config.EI = Map(config.gridpoints_y,
            config.r_T,
            config.r_F,
            config.r_M_later,
            N_init_trees = config.N_trees_arrival, 
            angleThreshold=config.AngleThreshold)
    print("Finished creating the map, now saving...")
    with open(filename, "wb") as EIfile:
        pickle.dump(config.EI, EIfile)
    print("... and plotting...")
    #config.EI.plot_agriculture(which="high", save=True)
    #config.EI.plot_agriculture(which="low", save=True)
    #config.EI.plot_agriculture(which="both", save=True)
    #config.EI.plot_TreeMap_and_hist(name_append="Tinit16e6", hist=True, save=True)
    #config.EI.plot_water_penalty()
    #config.EI.plotArea()

config.EI_triObject = mpl.tri.Triangulation(config.EI.points_EI_km[:,0], config.EI.points_EI_km[:,1], triangles = config.EI.all_triangles, mask=config.EI.mask)
print("Initialising Trees, ...")
config.EI.init_trees(config.N_trees_arrival)
print("Total Trees:", np.sum(config.EI.T_c))
print("Fraction of islands with forest: ",  np.sum([config.EI.A_c[i] for i in range(config.EI.N_c) if config.EI.T_c[i]>0])/ (np.sum(config.EI.A_c)))
config.EI.carrying_cap = deepcopy(config.EI.T_c)
config.well_suited_cells_tarrival= deepcopy(config.EI.well_suited_cells)
#config.tree_growth_poss = np.where(config.EI.carrying_cap>0)[0].astype(int)

config.F_pi_c_tarrival = deepcopy(config.EI.F_pi_c)

config.EI.get_P_G()

#EI_triangles = config.EI_triObject.get_masked_triangles()
#all_triangles = triObject.triangles

# Plot/Test the resulting Map.
#print("Test Map: ",config.EI.get_triangle_of_point([1e4, 1e4], config.EI_triObject))


print("################   DONE: MAP ###############")

#################################################
########   Folder and Run Prepa   ###############
#################################################
config.folder += "FullModel_grid"+str(config.gridpoints_y)+\
    "_gH1"+'%.0f' % (1000*(config.g_H1-1))+"e-3_"+\
    TreeRegrowthOption+"_"+\
    HighorLowFix+"_"+\
    TPrefOption+"_"+\
    PopulationStability+"_"+\
    AlphaSetting+\
    "_seed"+str(config.seed)+"/"




np.random.seed(config.seed)

##################################
###   PREPARE PENALTIES  #########
##################################
#WhichTrianglesToCalc_inds = np.arange(config.EI.N_c)
#config.MaxReachableTrees = np.max(np.dot(config.EI.tree_density, config.EI.TreeNeighbours_of_triangles[:,WhichTrianglesToCalc_inds]))
#availableAgric = config.EI.agric_yield * (config.EI.nr_highqualitysites + config.EI.nr_lowqualitysites - config.EI.agriculture)
#config.MaxYield = np.max(np.dot(availableAgric, config.EI.AgricNeighbours_of_triangles[:, WhichTrianglesToCalc_inds]))
#config.MaxYield = config.maxNeededAgric*2
#config.MaxReachableTrees = 45* config.tree_need_per_capita*config.max_pop_per_household_mean

################################
####     Run   Model        ####withTreegwithTreegrowthrowth
################################
def run():
    RUNSTART = time()
    ''' run the model specified in the config file'''
    # Save the current state of the implementation to the Subfolder
    Path(config.folder).mkdir(parents=True, exist_ok=True)
    Path(config.folder+"/USED_programs/").mkdir(parents=True, exist_ok=True)
    for file in ["FullModel.py","config.py", "agents.py", "CreateEI_ExtWaterExtAgricPaper.py", 
                "LogisticRegrowth.py", "observefunc.py", "update.py"]:
        shutil.copy(file, config.folder+"/USED_programs/")
    print("Working in ",config.folder,"; Python Scripts have been copied into subfolder USED_programs")
    
    print("################   RUN ###############")
    config.Array_pop_ct = np.array([[] for _ in range(config.EI.N_c)]).reshape((config.EI.N_c,0))
    config.Array_T_ct = np.array([[] for _ in range(config.EI.N_c)]).reshape((config.EI.N_c,0))
    config.Array_F_ct = np.array([[] for _ in range(config.EI.N_c)]).reshape((config.EI.N_c,0))
    # RUN

    config.EI.check_drought(config.t_arrival, config.droughts_RanoRaraku, config.EI_triObject)

    init_agents()
    
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(1,1,1, fc="gainsboro")
    observe(config.t_arrival, fig=fig, ax=ax, specific_ag_to_follow=None, save=True, data = None, ncdf=False, folder=config.folder, posSize_small=False, legend=True, cbar=[True, False, False])
    fig.tight_layout()
    plt.close()
    
    #observe(config.StartTime)
    for t in np.arange(config.t_arrival,config.t_end+1):#, num=config.N_timesteps):
        #if t==config.FishingTabooYear:
        #    config.MaxFisherAgents = config.NrFisherAgents[-1]

        update_time_step(t)  


        #################################
        #####   UPDATE ENVIRONMENT   ####
        #################################
        if TreeRegrowthOption=="withRegrowth": 
            regrow_update(config.tree_regrowth_rate)
        else:
            config.RegrownTrees = np.append(config.RegrownTrees, 0)
        
        agric_update()
        if TreeRegrowthOption=="withRegrowth": 
            popuptrees(t)
        else:
            config.TreesPoppedUp = np.append(config.TreesPoppedUp, 0)
        
        config.EI.check_drought(t, config.droughts_RanoRaraku, config.EI_triObject)
        
        if np.isnan(config.timeSwitchedMovingRad) and len(config.agents)>0 and np.sum(config.Array_pop_ct[:,-1]) >= config.pop_restrictedMoving :
            for ag in config.agents:
                ag.r_M=config.r_M_later
            config.timeSwitchedMovingRad=t
            print("Agents reduced their Moving Radius to ", config.r_M_later)
        
        ##############################
        #####  Plot / STORE   ########
        ##############################
        if len(config.agents)>0:
            if (t)%50==0:
                if len(config.agents)>0:
                    ag = config.agents[0]
                else:
                    ag = None
                fig = plt.figure(figsize=(16,8))
                ax = fig.add_subplot(1,1,1, fc="gainsboro")
                observe(t, fig=fig, ax=ax, specific_ag_to_follow=None, save=True, data = None, ncdf=False, folder=config.folder, posSize_small=False, legend=True, cbar=[True, True, True])
                fig.tight_layout()
                plt.close()
            #if (t+1)%50==0:
            #    plt.cla()
            #    _,ax = config.EI.plot_agricultureSites(ax=None, fig=None, save=False, CrossesOrFacecolor="Facecolor")
            #    ax.set_title("Time "+str(t+1))
            #    plt.savefig(config.folder+"AgricDensityMap_t"+str(t+1)+".png")
            #    plt.close()


    RUNEND = time()-RUNSTART
    print("END TIME- START TIME: ", RUNEND)
    print("Finished running Model in folder: ", config.folder)


        

if __name__ == "__main__":
    run()
    #plot_dynamics(config.N_timesteps)
    final_saving()
    #plot_statistics(config.folder)

    data = xr.open_dataset(config.folder+"Statistics.ncdf")
    plot_statistics(data, config.folder)

