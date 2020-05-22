import config

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pickle
from copy import copy

from observefunc import plot_movingProb


def calc_penalty(ag, WhichTrianglesToCalc_inds, t):  
    #if not ag.triangle_ind == config.EI.get_c_of_point([ag.x, ag.y], config.EI_triObject)[0]:
    #    print("ERRROR, triangle ind is not correct")
    
    C_T_ci_arr = config.EI.C_T_c[:,WhichTrianglesToCalc_inds]
    C_F_ci_arr = config.EI.C_F_c[:,WhichTrianglesToCalc_inds]
    
    ############################# 
    ## Tree PENALTY       #######
    ############################# 
    tr = np.dot(config.EI.T_c, C_T_ci_arr) 
    k_T = config.k_X(config.tr01, ag.tr99)
    P_T = config.logistic(tr, k_T, (config.tr01+ag.tr99)/2)
    NecessaryCond_T = np.array(tr>ag.tr99, dtype=np.uint8)
    P_T[np.where(NecessaryCond_T==False)]=1e15
   
    ############################# 
    ## Pop Densi PENALTY  #######
    #############################               
    pd=np.dot(config.EI.pop_c, C_F_ci_arr) / (ag.r_F**2 * np.pi) #* 1e6
    k_D = config.k_X(config.pd01,config.pd99)
    P_D = config.logistic(pd, k_D, (config.pd01+config.pd99)/2) #config.scaled_tanh(pop_density_atnb / config.MaxPopulationDensity, config.P50p, config.PkappaP)
    #NecessaryCond_D = pd < config.pd99 # smalle than Max!
    #if len(np.where(NecessaryCond_D==False))>0:
    #    P_D[np.where(NecessaryCond_D==False)]=1e10
    
    ############################# 
    ## Water     PENALTY  #######
    #############################
    # Water Penalty of all triangles in moving radius       
    P_W =config.EI.P_W[WhichTrianglesToCalc_inds]

    ############################# 
    ## Agriculture PENALTY  #####
    #############################
    # # MAY 10 Update:
    # reachableTotalAgricYield = np.dot(config.EI.agric_yield*(config.EI.nr_highqualitysites + config.EI.nr_lowqualitysites - config.EI.agriculture), C_F_ci_arr)  
    # reachableHighAgricYield = np.dot((config.EI.agric_yield==1)*(config.EI.nr_highqualitysites + config.EI.nr_lowqualitysites - config.EI.agriculture), C_F_ci_arr)  
    # y  = 1- np.sqrt(reachableHighAgricYield * reachableTotalAgricYield) / config.MaxYield
    # agriculture_penalty = config.scaled_tanh(y, config.P50a, config.PkappaA)
    # survivalCond_agric = np.array(reachableTotalAgricYield>ag.AgriNeed, dtype=np.uint8)
    # if config.FisherAgents < config.MaxFisherAgents: # still place for Fishers
    #     fishpossibility = np.dot(np.eye(1,config.EI.N_els, config.EI.AnakenaBeach_ind, dtype=np.uint8), C_F_ci_arr)
    #     agriculture_penalty[np.where(fishpossibility[0,:])] = - ag.treePref
    #     survivalCond_agric[np.where(fishpossibility[0,:])] = True

    # if len(np.where(survivalCond_agric ==False))>0:
    #     agriculture_penalty[np.where(survivalCond_agric==False)] = 1

    f_tot = np.dot(config.EI.F_pi_c*(config.EI.A_acres_c - config.EI.A_F_c), C_F_ci_arr)  
    k_F = config.k_X(config.f01, ag.f99)
    P_F_tot = config.logistic(f_tot, k_F,(config.f01+ ag.f99)/2 )
    f_well = np.dot((config.EI.F_pi_c==config.F_PI_well)*(config.EI.A_acres_c - config.EI.A_F_c), C_F_ci_arr)  
    P_F_well = config.logistic(f_well, k_F,(config.f01+ag.f99)/2 )
    P_F = 0.5*(P_F_well+P_F_tot)
    NecessaryCond_F = np.array(f_tot>ag.f99, dtype=np.uint8)
    P_F[np.where(NecessaryCond_F==False)]=1e15
    #fishingcells = np.array([i for i in WhichTrianglesToCalc_inds if i in config.EI.C_F_cAnakena])
    #fishingcells = np.where(np.dot(config.EI.C_F_cAnakena, C_F_ci_arr))[0]
    if config.FisherAgents<config.MaxFisherAgents:
        fishingcells = np.isin(WhichTrianglesToCalc_inds, config.EI.C_F_cAnakena)
        P_F[fishingcells] = -1
   
    ################################
    #####     MAP PENALTY   ########
    ################################
    P_G = config.EI.P_G[WhichTrianglesToCalc_inds]
    #slopes_cond = config.EI.slopes_cond[WhichTrianglesToCalc_inds]

    
    ################################
    #####     Tot Penalty   ########
    ################################    # SUM OF ALPHA Should BE 1 then Max and Min Prob are 1 and 0!
    P_tot = ( ag.alpha_W * P_W +  ag.alpha_T_prime* P_T + ag.alpha_D * P_D + ag.alpha_F_prime * P_F + ag.alpha_G * P_G)

    #total_penalties+= ((1-maske)*1).clip(0,1)
    return  P_tot, ag.alpha_T_prime,  ag.alpha_F_prime, [P_W, P_G, P_T, P_F, P_D], NecessaryCond_T, NecessaryCond_F



def move(ag, t, FirstSettlementAnakenaTriangles=[False, np.array([])]):
    
    # CLEAR YOUR OLD SPACE!
    config.EI.agNr_c[ag.c_i] -=1
    config.EI.pop_c[ag.c_i] -=ag.pop
    for site in ag.A_F_i:
        config.EI.A_F_c[site] -=1
    ag.A_F_i=np.array([]).astype(int)
    ag.F_PI_i = np.array([])
    if ag.fisher: 
        config.FisherAgents-=1
        ag.fisher = False
        ag.calc_new_tree_pref()
        ag.calc_F_Req()
        ag.calc_T_Req()


    if ag.r_M < 100:
        WhichTrianglesToCalc_inds = np.where(config.EI.C_Mlater_c[:,ag.c_i])[0]
    else:
        WhichTrianglesToCalc_inds = np.arange(config.EI.N_c)
    if FirstSettlementAnakenaTriangles[0]:
        WhichTrianglesToCalc_inds = FirstSettlementAnakenaTriangles[1]
    P_tot, ag.alpha_T_prime, ag.alpha_F_prime,[P_W, P_G, P_T, P_F, P_D], NecessaryCond_T, NecessaryCond_F = calc_penalty(ag, WhichTrianglesToCalc_inds,t)
    
    Pr_c= np.exp(-config.gamma*P_tot)#(1-total_penalties))



    ####################
    ##   MOVE   #######
    ####################
    previous_triangle = copy(ag.c_i)
    previous_x = copy(ag.x)
    previous_y = copy(ag.y)
    
    if any(Pr_c>0):
        Pr_c=Pr_c/np.sum(Pr_c)

        # OPTION 1 choose one random of the 5 lowest penalties
        # OPtion 2 choose according to p.
        # ag.triangle_ind = ag.mv_inds[np.random.choice(np.argpartition(probabilities, -config.NrOfEquivalentBestSitesToMove)[-config.NrOfEquivalentBestSitesToMove:])]
        #ag.triangle_ind = ag.mv_inds[np.argmax(probabilities)]
        ag.c_i = np.random.choice(WhichTrianglesToCalc_inds, p= Pr_c)
    else:
        #print("Agent", ag.index, " can't find allowed space. All probs ==0.")
        ag.c_i = np.random.choice(WhichTrianglesToCalc_inds)

    ag.vec_c_i = config.EI.vec_c[ag.c_i]
    config.EI.agNr_c[ag.c_i] +=1
    config.EI.pop_c[ag.c_i] += ag.pop

    ###################################################
    # Within the triangle search for random position! #
    ###################################################
    A,B,C =  config.EI.points_EI_km[config.EI.EI_triangles[ag.c_i]]
    # Draw points in trapez:
    bary1 = np.random.random()
    bary2= np.random.random()
    # random point in trapezoid. If in other half of trapezoid (not in triangle), just mirror along the diagonal
    p = A+ bary1 * (B-A) + bary2*(C-A)  
    if bary1+bary2 <1:
        p = p
    else:
        # Mirror point:
        M = C+(B-C)*0.5#C + np.dot((p-C), C-B) * (C-B)/(np.linalg.norm(C-B)**2)
        p = p+2*(M-p)
    ag.x = p[0] #ag.triangle_midpoint[0]
    ag.y = p[1] #ag.triangle_midpoint[1]


    if not ag.c_i == config.EI.get_c_of_point([ag.x, ag.y], config.EI_triObject)[0]:
        print("ERRROR, triangle ind is not correct", ag.index)
    ag.penalty = P_tot[np.where(WhichTrianglesToCalc_inds==ag.c_i)[0]]



    ###########################################
    ####   PLOT PENALTIES FOR A FEW AGENTS   ##
    ###########################################
    if ag.index in [0,5,10,50,70, 80,100,150, 200,250,500,800]: #and (not FirstSettlementAnakenaTriangles[0]):
        #water_penalty, tree_penalty, pop_density_penalty, agriculture_penalty, map_penalty = penalties
        #slopes_cond, population_cond, survivalCond_agric, survivalCond_tree = masken
        #allowed_colors = np.array([survivalCond_agric,survivalCond_tree, np.array(population_cond),np.array(slopes_cond)],dtype=np.uint8)
        print("Saving Agent ", ag.index,"'s penalty ...", end="")
        ToSavePenalties = {
            "AG_total_penalties":P_tot,
            "AG_water_penalties":P_W,
            "AG_tree_penalty":P_T,
            "AG_pop_density_penalty":P_D,
            "AG_agriculture_penalty":P_F,
            "AG_map_penalty":P_G,
            "AG_nosettlement_zones": NecessaryCond_T*NecessaryCond_F, # 0 if P=infty, 1 if P finite.
            "AG_MovingProb":Pr_c,
            "AG_Pos": [previous_x, previous_y],
            "AG_NewPos": [ag.x, ag.y],
            "AG_InitialTriangle":previous_triangle,
            "AG_NewPosTriangle":ag.c_i,
            "AG_MovingRadius":ag.r_M,
            "AG_TPref":ag.T_Pref_i,
            "AG_pop":ag.pop,
            "AG_H":ag.H_i,
            "t":t,
            "AG_index":ag.index,
        }
        filename = config.folder+"Penalties_AG"+str(ag.index)+"_t="+str(t)
        with open(filename, "wb") as AGSavefile:
            pickle.dump(ToSavePenalties, AGSavefile)
        #print("Plotting...", end="")
        #plot_movingProb(ToSavePenalties)
        #plt.close()
        print("DONE")
    return 
