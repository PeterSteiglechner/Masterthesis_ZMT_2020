import config

import numpy as np 
import matplotlib.pyplot as plt 

import sys
sys.path.append("/home/peter/EasterIslands/Code/Triangularisation/")
#from CreateMap_class import Map


def regrow_update(g0):
    #def logistic(n):
    #    if config.EI.carrying_cap[n] == 0:
    #        return 0
    #    else:
    #        return (g0 * (1 - ( config.EI.tree_density[n]/config.EI.carrying_cap[n]))
    ##d = (config.EI.carrying_cap - config.EI.tree_density)
    #g = np.array([ logistic(n) if config.EI.agriculture[n]==0 else 0 for n in range(config.EI.N_els)])
    #g = np.zeros(config.EI.N_els)
    inds = np.where(    (config.EI.agriculture==0)* (config.EI.carrying_cap>0) * (config.EI.tree_density>0) * (config.EI.tree_density<config.EI.carrying_cap))[0]
    g = g0 * (1 - ( config.EI.tree_density[inds]/config.EI.carrying_cap[inds]))
    #print("g: ", np.max(g), np.min(g), np.mean(g), np.nonzero(g))
    #grow = np.random.random(len(inds)) < g
    #print("Trees regrowing are ", np.nonzero(grow))
    #inds = np.array([ i  for i in range(0,config.EI.N_els) if (not (grow[i]==0))], dtype=int)
    #inds = inds[np.where(grow)]
    #for i in inds:
    #    config.EI.tree_density[i]= min(config.EI.carrying_cap[i], config.EI.tree_density[i]+np.round(g[i],0)).astype(int)
    regrownTrees = np.round(g*config.EI.tree_density[inds],0).astype(int)
    config.EI.tree_density[inds]=(config.EI.tree_density[inds]+regrownTrees).clip(max = config.EI.carrying_cap[inds]).astype(int)
    #print("#new nr of trees: ", np.sum(config.EI.tree_density))
    config.TreeRegrowth.append(np.sum(regrownTrees))
    return 

def agric_update():
    config.EI.agric_yield = config.initial_agric_yield
    notreecells = np.where(config.EI.tree_density==0)[0].astype(int)
    #config.EI.treeless_years[notreecells]+=1
    # For all cells with 20 years without trees, decrease soil quality to Eroded Soil yield or even lower
    #if config.EI.treeless_years[i]>config.YearsBeforeErosionDegradation else config.EI.agric_yield[i] 
    config.EI.agric_yield[notreecells] =  np.array([np.minimum(config.EI.agric_yield[i], config.ErodedSoilYield) for i in notreecells])
    #for n,triang in enumerate(config.EI.EI_triangles):
    #    if 
   
    return 

def popuptrees(t):
    if t>config.StartTime+config.tree_pop_timespan:
        #no_agric = np.where(config.EI.agriculture==0)

        to_pop_up = config.tree_growth_poss[np.where(config.EI.tree_density[config.tree_growth_poss]==0)[0]]#.astype(int)

        no_agric = to_pop_up[np.where(config.Array_agriculture[to_pop_up,-config.tree_pop_timespan:].sum(axis=1) == 0)[0]]

        
        #inds, counts = np.unique(np.concatenate([no_agric, to_pop_up,tree_growth_poss]), return_counts=True)
        
        #where_to_pop = inds[counts>2]
        where_to_pop = no_agric

        config.EI.tree_density[where_to_pop] = (config.tree_pop_percentage * config.EI.carrying_cap[where_to_pop]).astype(int)
        config.TreePopUp.append(np.sum(config.EI.tree_density[where_to_pop]).astype(int))
    else:
        config.TreePopUp.append(0)
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
