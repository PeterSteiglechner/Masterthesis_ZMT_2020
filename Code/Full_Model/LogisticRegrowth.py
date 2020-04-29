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
    g = np.array([ logistic(n) if config.EI.agriculture[n]==False else 0 for n in range(config.EI.N_els)])
    #print("g: ", np.max(g), np.min(g), np.mean(g), np.nonzero(g))
    grow = np.random.random(config.EI.N_els) < g
    #print("Trees regrowing are ", np.nonzero(grow))
    inds = [ i  for i in range(0,config.EI.N_els) if (not (grow[i]==0))]
    for i in inds:
        config.EI.tree_density[i]= min(config.EI.carrying_cap[i], config.EI.tree_density[i]+np.round(g[i], 0))
    #print("#new nr of trees: ", np.sum(config.EI.tree_density))
    return 

def agric_update():
    notreecells = np.where(config.EI.tree_density==0)[0]
    config.EI.treeless_years[notreecells]+=1
    # For all cells with 20 years without trees, decrease soil quality to Eroded Soil yield or even lower
    config.EI.agric_yield[notreecells] =  np.array([np.min(config.EI.agric_yield[i], config.ErodedSoilYield) if config.EI.treeless_years[i]>config.YearsBeforeErosionDegradation else config.EI.agric_yield[i] for i in notreecells])
    #for n,triang in enumerate(config.EI.EI_triangles):
    #    if 
    return 

def popuptrees():
    # SET treeless_years to 0
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
