
#!/usr/bin/env python

# PETER STEIGLECHNER, 10.3.2020

# Take config file, and save that

import config
import xarray as xr 
import numpy as np
import pandas as pd 



def final_saving():
    ''' Saves the aggregated Numbers to a ncdf file'''
    times = np.arange(config.t_arrival,config.t_end+1)
    agent_indices = np.arange(config.index_count)

    #possize = np.array(config.agentStats_t)
    PosAgents = np.empty([config.index_count, 2,len(times)])
    PosAgents[:] = np.nan 
    SizeAgents = np.empty([config.index_count, len(times)])
    SizeAgents[:] = np.nan
    TreePrefAgents = np.empty([config.index_count, len(times)])
    TreePrefAgents[:]=np.nan
    for t in range(len(times)):
        if not config.Array_N_agents[t]==0:
            SizeAgents[config.agentStats_t[t][:,0].astype(int), t] = config.agentStats_t[t][:,3]
            PosAgents[config.agentStats_t[t][:,0].astype(int), :,t] = config.agentStats_t[t][:,1:3]
            TreePrefAgents[config.agentStats_t[t][:,0].astype(int),t] = config.agentStats_t[t][:,4]
    ds = xr.Dataset(
        {
            "total_agents":("time", config.Array_N_agents),
            #"total_population": ("time", config.A),
            "total_excessdeaths": ("time", config.nr_excessdeaths),
            "total_excessbirths": ("time", config.nr_excessbirths),
            #"total_trees": ("time", config.nr_trees),
            "T_ct":(("triangles","time"), config.Array_T_ct),
            "F_ct":(("triangles","time"), config.Array_F_ct),
            "pop_ct":(("triangles","time"), config.Array_pop_ct),
            "happyMeans":(("time",),config.happyMeans),
            "happyStd":(("time",),config.happyStd),
            "treeFills":(("time",),config.treeFills),
            "farmingFills":(("time",),config.farmingFills),
            "SizeAgents":(('index', 'time'),SizeAgents),
            "PosAgents":(('index', '2d', "time"), PosAgents),
            "TreePrefAgents":(('index', "time"), TreePrefAgents),
            "Penalty_Mean":(('time'), config.Penalty_mean),
            "Penalty_Std":(('time'), config.Penalty_std),
            "FishersPop":(("time"), config.NrFisherAgents),
            "GardenFraction":(("time"), config.GardenFraction),
            "Fraction_eroded":(("time"), config.Fraction_eroded),
            #"agents":(("ag_ind", "properties", "time"), t1)
            "Nr_Moves":(("time"), config.moves),
            "TreeRegrowth":(("time"), config.RegrownTrees),
            "TreePopup":(("time"), config.TreesPoppedUp),
        },
        {
            "time": times,
            "index":agent_indices,
            "triangles":np.arange(config.EI.N_c),
            "2d":["x","y"],
        }#, "ag_ind": indices_agents, "properties": properties},
    )
    if len(config.fire)>0:
        ds.attrs['firesLoc'] = np.array(config.fire)[:,0]
        ds.attrs['firesSize'] = np.array(config.fire)[:,1]
        ds.attrs['firesTime'] = np.array(config.fire)[:,2]
    else:
        ds.attrs['firesLoc'] = np.array([])
        ds.attrs['firesSize'] = np.array([])
        ds.attrs['firesTime'] = np.array([])



    params_moving={
        'r_M_init': config.r_M_init,
        'r_M_later':config.r_M_later,
        'pop_restrictedMoving': config.pop_restrictedMoving,
        'w01': config.w01, 
        'w99': config.w99, 
        'el01': config.el01, 
        'el99': config.el99, 
        'sl01': config.sl01, 
        'sl99': config.sl99, 
        'tr01': config.tr01, 
        'f01': config.f01, 
        'pd01': config.pd01, 
        'pd99': config.pd99, 
        'alpha_W':config.alpha_W,
        'alpha_T':config.alpha_T,
        'alpha_D':config.alpha_D,
        'alpha_F':config.alpha_F,
        'alpha_G':config.alpha_G,
        'gamma':config.gamma,
    }
    for key in params_moving.keys():
        ds.attrs[key]=params_moving[key]

    ds.attrs["timeSwitchedMovingRad"] = config.timeSwitchedMovingRad


    params_households = {
        'r_T': config.r_T,
        'r_F': config.r_F,
        'F_Req_pP': config.F_Req_pP,
        'T_Req_pP': config.T_Req_pP,
        'T_Pref_max': config.T_Pref_max,
        "T_Pref_min":config.T_Pref_min,
       'xi_T_Pref': config.xi_T_Pref,
        "T_Pref_fisher_min":config.T_Pref_fisher_min,
        'MaxFisherAgents': config.MaxFisherAgents, 
        "g_H1":config.g_H1, 
        "H_equ":config.H_equ,
        "SplitPop": config.SplitPop,
        "pop_agent_max_mean":config.pop_agent_max_mean,
        "pop_agent_max_std":config.pop_agent_max_std,
        "pop_agent_min":config.pop_agent_min,
        "g_shape":config.g_shape,
        "g_scale":config.g_scale,
    }
    for key in params_households.keys():
        ds.attrs[key]=params_households[key]
    
    

    params_init={
        'N_agents_arrival': config.N_agents_arrival,
        'Pop_arrival': config.Pop_arrival,
        "r_M_arrival":config.r_M_arrival,
    }
    for key in params_init.keys():
        ds.attrs[key]=params_init[key]
    
    params_run={
        'updatewithreplacement': np.uint8(config.updatewithreplacement),
        'seed': config.seed,
        "t_end":config.t_end,
        'folder': config.folder,
    }
    for key in params_run.keys():
        ds.attrs[key]=params_run[key]
    


    params_map = {
    'gridpoints_y': config.gridpoints_y,
    'AngleThreshold': config.AngleThreshold ,
    'N_trees_arrival': config.N_trees_arrival,
    'km2_to_acre': config.km2_to_acre,
    'tree_regrowth_rate': config.tree_regrowth_rate,
    "F_PI_poor":config.F_PI_poor,
    "F_PI_eroded":config.F_PI_eroded,
    "F_PI_well":config.F_PI_well,
    #"YearsBeforeErosionDegradation":config.YearsBeforeErosionDegradation,
    "tree_pop_percentage":config.tree_pop_percentage,
    "barrenYears_beforePopUp":config.barrenYears_beforePopUp,
    "drought_RanoRaraku_1_start":config.droughts_RanoRaraku[0][0],
    "drought_RanoRaraku_1_end":config.droughts_RanoRaraku[0][1],
    "drought_RanoRaraku_2_start":config.droughts_RanoRaraku[1][0],
    "drought_RanoRaraku_2_end":config.droughts_RanoRaraku[1][1],
    }
    for key in params_map.keys():
        ds.attrs[key]=params_map[key]
    
    #config.TreeDensityConditionParams = {'minElev':10, 'maxElev': 400,'maxSlope': 7}
    for key in config.TreeDensityConditionParams.keys():
        ds.attrs["TreeDensityConditionParams_"+key] = config.TreeDensityConditionParams[key]
    
    ds.to_netcdf(path=config.folder+"Statistics.ncdf")
    print("SAVED NCDF TO ",config.folder+"Statistics.ncdf" )
    return 

'''
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
                'Midpoints_x': (midpoints[:,0]),
                'Midpoints_y': (midpoints[:,1]),
                'TreeDensity': (data)
            }, 
            #columns=['Midpoints_x', 'Midpoints_y','TreeDensity']
        )
        df_env = df_env.set_index(['Midpoints_x', 'Midpoints_y'])
        #df.attrs["Time"]=t
        df_env.to_csv(config.folder+"AgentsStats_t="+str(t)+"_env.csv")
    
    # TODO! Environment as DataSet!
    return 
'''
