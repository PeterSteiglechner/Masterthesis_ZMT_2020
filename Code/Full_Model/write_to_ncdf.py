#!/usr/bin/env python

# PETER STEIGLECHNER, 10.3.2020

# Take config file, and save that

import config
import xarray as xr 
import numpy as np
import pandas as pd 



def final_saving():
    ''' Saves the aggregated Numbers to a ncdf file'''
    times = np.arange(config.StartTime,config.EndTime+1)
    agent_indices = np.arange(config.index_count)

    #possize = np.array(config.agents_pos_size)
    PosAgents = np.empty([config.index_count, 2,len(times)])
    PosAgents[:] = np.nan 
    SizeAgents = np.empty([config.index_count, len(times)])
    SizeAgents[:] = np.nan
    for t in range(len(times)):
        if not config.nr_agents[t]==0:
            SizeAgents[config.agents_pos_size[t][:,0].astype(int), t] = config.agents_pos_size[t][:,3]
            PosAgents[config.agents_pos_size[t][:,0].astype(int), :,t] = config.agents_pos_size[t][:,1:3]

    ds = xr.Dataset(
        {
            "total_agents":("time", config.nr_agents),
            "total_population": ("time", config.nr_pop),
            "total_deaths": ("time", config.nr_deaths),
            "total_trees": ("time", config.nr_trees),
            "tree_density":(("triangles","time"), config.Array_tree_density),
            "agriculture":(("triangles","time"), config.Array_agriculture),
            "populationOccupancy":(("triangles","time"), config.Array_populationOccupancy),
            "happyFraction":(("time",),config.nr_happyAgents),
            "SizeAgents":(('index', 'time'),SizeAgents),
            "PosAgents":(('index', '2d', "time"), PosAgents),
            "Penalty_Mean":(('time'), config.Penalty_mean),
            "Penalty_Std":(('time'), config.Penalty_std),
            "FishersPop":(("time"), config.NrFisherAgents)
            #"agents":(("ag_ind", "properties", "time"), t1)
        },
        {
            "time": times,
            "index":agent_indices,
            "triangles":np.arange(config.EI.N_els),
            "2d":["x","y"],
        }#, "ag_ind": indices_agents, "properties": properties},
    )
    if len(config.fire)>0:
        ds.attrs['firesLoc'] = np.array(config.fire)[:,0]
        ds.attrs['firesSize'] = np.array(config.fire)[:,1]
        ds.attrs['firesTime'] = np.array(config.fire)[:,2]

    params_households = {
        'MaxSettlementSlope': config.MaxSettlementSlope,
        #'MaxSettlementElev': config.MaxSettlementElev,
        #'MinSettlementElev': config.MinSettlementElev,
        'SweetPointSettlementElev':config.SweetPointSettlementElev,
        'max_pop_per_household_mean': config.max_pop_per_household_mean,
        'max_pop_per_household_std': config.max_pop_per_household_std,
        'MaxPopulationDensity': config.MaxPopulationDensity,
        #'MaxAgricPenalty': config.MaxAgricPenalty,
        'tree_need_per_capita': config.tree_need_per_capita,
        'MinTreeNeed': config.MinTreeNeed,
        'BestTreeNr_forNewSpot': config.BestTreeNr_forNewSpot,
        #'Nr_AgricStages': config.Nr_AgricStages,
        #'dStage': config.dStage,
        'agricYield': config.agricYield_need_per_Capita,
        #'treePref_decrease_per_year': config.treePref_decrease_per_year, 
        #'treePref_change_per_BadYear': config.treePref_change_per_BadYear,
        #'HowManyDieInPopulationShock': config.HowManyDieInPopulationShock,
        #'FractionDeathsInPopShock':config.FractionDeathsInPopShock,
        "LowerLimit_PopInHousehold":config.LowerLimit_PopInHousehold,
        'tree_search_radius': config.params['tree_search_radius'], 
        'agriculture_radius': config.params['agriculture_radius'], 
        'moving_radius': config.params['moving_radius'], 
        'reproduction_rate': config.params['reproduction_rate'], 
        'alpha_w':config.alpha_w,
        'alpha_t':config.alpha_t,
        'alpha_p':config.alpha_p,
        'alpha_a':config.alpha_a,
        'alpha_m':config.alpha_m,
        'PenalToProb_Prefactor':config.PenalToProb_Prefactor,
        "MaxFisherAgents":config.MaxFisherAgents,
        #"NrOfEquivalentBestSitesToMove":config.NrOfEquivalentBestSitesToMove,
        }
    
    for key in params_households.keys():
        ds.attrs[key]=params_households[key]
    
    params_init={
        'N_agents': config.N_agents,
        'N_trees': config.N_trees,
        'init_pop': config.init_pop,
        'init_TreePreference': config.init_TreePreference,
        #'blabla': config.index_count=0
    }
    for key in params_init.keys():
        ds.attrs[key]=params_init[key]
    
    params_run={
        'updatewithreplacement': np.uint8(config.updatewithreplacement),
        'N_timesteps': config.N_timesteps,
        'seed': config.seed,
        'folder': config.folder,
    }
    for key in params_run.keys():
        ds.attrs[key]=params_run[key]
    
    params_map = {
    'gridpoints_y': config.gridpoints_y,
    'AngleThreshold': config.AngleThreshold ,
    'km2_to_acre': config.km2_to_acre,
    'tree_regrowth_rate': config.tree_regrowth_rate,
    "UpperLandSoilQuality":config.UpperLandSoilQuality,
    "ErodedSoilYield":config.ErodedSoilYield,
    #"YearsBeforeErosionDegradation":config.YearsBeforeErosionDegradation,
    "tree_pop_percentage":config.tree_pop_percentage,
    "tree_pop_timespan":config.tree_pop_timespan,
    }
    for key in params_map.keys():
        ds.attrs[key]=params_map[key]
    
    #config.TreeDensityConditionParams = {'minElev':10, 'maxElev': 400,'maxSlope': 7}
    for key in config.TreeDensityConditionParams.keys():
        ds.attrs["TreeDensityConditionParams_"+key] = config.TreeDensityConditionParams[key]
    #config.AgriConds={'minElev':20,'maxElev_highQu':250,
    #    'maxSlope_highQu':3.5,'maxElev_lowQu':380,'maxSlope_lowQu':6,
    #    'MaxWaterPenalty':300,}
    #for key in config.AgriConds.keys():
    #    ds.attrs["AgricConds_"+key] = config.AgriConds[key]
    
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
