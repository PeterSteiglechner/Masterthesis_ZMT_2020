import config

import numpy as np 
import matplotlib.pyplot as plt 

import sys
sys.path.append("/home/peter/EasterIslands/Code/Triangularisation/")
#from CreateMap_class import Map


def regrow_update(g0):
    # cells with no agriculture, no carrying cap, no T_c but carrying_cap>0
    inds = np.where(    (config.EI.A_F_c==0)* (config.EI.carrying_cap>0) * (config.EI.T_c>0) * (config.EI.T_c<config.EI.carrying_cap))[0]
    # Growth Rate 
    g = g0 * (1 - ( config.EI.T_c[inds]/(config.EI.carrying_cap[inds])))    #(1-config.EI.agriculture[inds]/config.EI.EI_triangles_areas[inds])
    regrownTrees = np.round(g*config.EI.T_c[inds],0).astype(int)
    config.EI.T_c[inds]=(config.EI.T_c[inds]+regrownTrees).clip(max = config.EI.carrying_cap[inds]).astype(int)
    config.RegrownTrees = np.append(config.RegrownTrees, np.sum(regrownTrees))
    return 

def agric_update():
    # set all cells to well-suited
    config.EI.F_pi_c = config.F_pi_c_tarrival
    config.EI.well_suited_cells = config.well_suited_cells_tarrival
    # check where no trees are present and the cell is well-suited
    cells_wellsuited_noTrees = np.where((config.EI.T_c==0)*(config.EI.F_pi_c==config.F_PI_well))[0].astype(int)
    # set these to eroded soil
    config.EI.F_pi_c[cells_wellsuited_noTrees] =  config.F_PI_eroded
    #np.array([np.minimum(config.EI.F_pi_c[i], config.F_PI_eroded) for i in notreecells])

    config.EI.eroded_well_suited_cells[cells_wellsuited_noTrees] = config.EI.A_acres_c[cells_wellsuited_noTrees]
    config.EI.well_suited_cells[cells_wellsuited_noTrees]= 0
    #config.poorly_suited_cells= np.where(config.EI.F_pi_c==config.F_PI_poor)[0]
    #config.eroded_well_suited_cells= np.where(config.EI.F_pi_c==config.F_PI_eroded)[0]
    return 

def popuptrees(t):
    if t>config.t_arrival+config.barrenYears_beforePopUp:
        # where no trees are present
        to_pop_up = np.where(config.EI.T_c==0)[0]
        #        #config.tree_growth_poss[np.where(config.EI.T_c[config.tree_growth_poss]==0)[0]]#.astype(int)
        # where no agriculture was present for treePop
        no_agric = to_pop_up[np.where(config.Array_F_ct[to_pop_up,-config.barrenYears_beforePopUp:].sum(axis=1) == 0)[0]]
        # these are all the indices where trees will pop
        where_to_pop = no_agric

        config.EI.T_c[where_to_pop] = (config.tree_pop_percentage * config.EI.carrying_cap[where_to_pop]).astype(int)
        config.TreesPoppedUp = np.append(config.TreesPoppedUp, np.sum(config.EI.T_c[where_to_pop]).astype(int))
    else:
        config.TreesPoppedUp = np.append(config.TreesPoppedUp, 0)
    return 


if __name__ == "__main__":
    pass