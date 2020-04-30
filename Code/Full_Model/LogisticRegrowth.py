import config

import numpy as np 
import matplotlib.pyplot as plt 

import sys
sys.path.append("/home/peter/EasterIslands/Code/Triangularisation/")
#from CreateMap_class import Map


def regrow_update(g0):
    def logistic(n):
        if config.EI.carrying_cap[n] == 0:
            return 0
        else:
            return (g0 * (1 - ( config.EI.tree_density[n]/config.EI.carrying_cap[n])))
    #d = (config.EI.carrying_cap - config.EI.tree_density)
    g = np.array([ logistic(n) if config.EI.agriculture[n]==0 else 0 for n in range(config.EI.N_els)])
    #print("g: ", np.max(g), np.min(g), np.mean(g), np.nonzero(g))
    grow = np.random.random(config.EI.N_els) < g
    #print("Trees regrowing are ", np.nonzero(grow))
    #inds = np.array([ i  for i in range(0,config.EI.N_els) if (not (grow[i]==0))], dtype=int)
    inds = np.where(grow)[0].astype(int)
    #for i in inds:
    #    config.EI.tree_density[i]= min(config.EI.carrying_cap[i], config.EI.tree_density[i]+np.round(g[i],0)).astype(int)
    config.EI.tree_density[inds]= np.minimum(config.EI.carrying_cap[inds], config.EI.tree_density[inds]+np.round(g[inds]*config.EI.tree_density[inds],0)).astype(int)
    #print("#new nr of trees: ", np.sum(config.EI.tree_density))
    return 

def agric_update():
    notreecells = np.where(config.EI.tree_density==0)[0].astype(int)
    #config.EI.treeless_years[notreecells]+=1
    # For all cells with 20 years without trees, decrease soil quality to Eroded Soil yield or even lower
    #if config.EI.treeless_years[i]>config.YearsBeforeErosionDegradation else config.EI.agric_yield[i] 
    config.EI.agric_yield[notreecells] =  np.array([np.minimum(config.EI.agric_yield[i], config.ErodedSoilYield) for i in notreecells])
    #for n,triang in enumerate(config.EI.EI_triangles):
    #    if 
    
    return 

def popuptrees(t):
    if t>config.StartTime+config.tree_pop_timespan+1:
        #no_agric = np.where(config.EI.agriculture==0)
        no_agric = np.where(config.Array_agriculture[:,-config.tree_pop_timespan:].sum(axis=1) == 0)[0].astype(int)

        to_pop_up = np.where(config.EI.tree_density==0)[0].astype(int)
        tree_growth_poss = np.where(config.EI.carrying_cap>0)[0].astype(int)
        
        inds, counts = np.unique(np.concatenate([no_agric, to_pop_up,tree_growth_poss]), return_counts=True)
        
        where_to_pop = inds[counts>2]

        config.EI.tree_density[where_to_pop] = config.tree_pop_percentage * config.EI.carrying_cap[where_to_pop]
    return 


if __name__ == "__main__":
    pass
    #from CreateEI_ExtWater import Map

    #config.TreeDensityConditionParams = {'minElev':0, 'maxElev':430,'maxSlope':7}


    #config.EI = Map(50, N_init_trees= 1e6, verbose=True)
    #config.EI.plot_TreeMap_and_hist(name_append = -1)

    #config.EI.init_trees(3e5)
    #for i in range(100):
    #    if i%5==0:
    #        config.EI.plot_TreeMap_and_hist(name_append = i)
    #    regrow_update(1)    
