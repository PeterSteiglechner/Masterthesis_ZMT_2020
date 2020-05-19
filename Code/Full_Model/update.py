
import numpy as np
from time import time
from copy import copy
import config
from moving import calc_penalty, move

# UPDATE
def update_time_step(t):
    ''' update one time step, i.e. perform update_single_agent() N_agents times'''
    start_step = time()
    N_agents = len(config.agents)
    if not N_agents==0:

        if config.updatewithreplacement==False:
            # i.e. Every Agent gets updated once per timestep
            #order = np.arange(0,N_agents), N_agents, replace=False).astype(int)
            agents_order = np.random.choice(config.agents, N_agents, replace=False)
            for selected_agent in agents_order:
                update_single_agent(selected_agent,t)

        elif config.updatewithreplacement==True:
            selected_agent = np.random.choice(config.agents)
            update_single_agent(selected_agent,t)
    
    ######## STORE ##############    
    config.Array_N_agents= np.append(config.Array_N_agents, len(config.agents))
    #config.nr_trees.append(np.sum(config.EI.tree_density))
    #config.nr_pop.append(np.sum([ag.pop for ag in config.agents]))
    config.nr_excessdeaths = np.append(config.nr_excessdeaths,config.excess_deaths)
    config.excess_deaths = 0
    config.nr_excessbirths= np.append(config.nr_excessbirths,config.excess_births)
    config.excess_births = 0
    # config.fires (updated during update)
    #config.agents_stages.append(np.histogram([ag.stage for ag in config.agents], bins=np.arange(config.Nr_AgricStages+1))[0])
    #config.TPrefs.append(np.histogram([ag.treePref for ag in config.agents], bins=np.linspace(config.MinTreeNeed, config.init_TreePreference, num=10))[0])

    agentStats = [np.array([int(ag.index), ag.x, ag.y, ag.pop, ag.T_Pref_i]) for ag in config.agents]
    if len(config.agents)==0:
        agentStats = [np.array([np.nan, np.nan, np.nan, np.nan])]
    config.agentStats_t.append(np.array(agentStats))#_t, np.array(agentStats).reshape(5,1), axis=1)


    # ENVIRONMENT:
    #config.nr_agricultureSites = np.append(config.nr_agricultureSites, np.sum(config.EI.A_F_c))
    if not len(config.agents)==0:
        happys_inclDead = np.zeros(config.index_count)
        happys_inclDead[0:len(config.agents)] = np.array([ag.h_i for ag in config.agents])
        happyMean= np.mean(happys_inclDead)
        config.happyMeans = np.append(config.happyMeans, happyMean)
        config.happyStd = np.append(config.happyStd, np.std(happys_inclDead))
        config.treeFills = np.append(config.treeFills, np.sum([ag.tree_fill<1 for ag in config.agents]))
        config.farmingFills = np.append(config.farmingFills, np.sum([ag.farming_fill<1 for ag in config.agents]))
        config.Penalty_mean = np.append(config.Penalty_mean, (np.mean([ag.penalty if ag.penalty<100 else 1 for ag in config.agents])) )
        config.Penalty_std = np.append(config.Penalty_std, (np.std([ag.penalty if ag.penalty<100 else 1 for ag in config.agents])))
        config.NrFisherAgents = np.append(config.NrFisherAgents, config.FisherAgents)
        config.GardenFraction = np.append(config.GardenFraction, np.mean([np.sum(ag.F_PI_i==config.F_PI_poor)/max(len(ag.A_F_i),1) for ag in config.agents]))
    else:
        config.happyMeans = np.append(config.happyMeans,0)
        config.happyStd = np.append(config.happyStd,0)
        config.treeFills = np.append(config.treeFills, 0)
        config.farmingFills = np.append(config.farmingFills, 0)
        config.Penalty_std = np.append(config.Penalty_std,0)
        config.Penalty_mean = np.append(config.Penalty_mean,0)
        config.NrFisherAgents = np.append(config.NrFisherAgents, 0)
        config.GardenFraction =  np.append(config.GardenFraction, 0)
    config.moves = np.append(config.moves, config.move)
    config.move=0

    config.Fraction_eroded = np.append(config.Fraction_eroded, np.sum(config.EI.F_pi_c==config.F_PI_eroded) /np.sum(config.F_pi_c_tarrival==config.F_PI_well))
    config.Array_T_ct = np.concatenate((config.Array_T_ct, config.EI.T_c.reshape((config.EI.N_c,1)).astype(np.int32)), axis=1).astype(np.int32) 
    config.Array_F_ct = np.concatenate((config.Array_F_ct, config.EI.A_F_c.reshape((config.EI.N_c,1)).astype(np.int32)), axis=1).astype(np.int32)
    config.Array_pop_ct = np.concatenate((config.Array_pop_ct, config.EI.pop_c.reshape((config.EI.N_c,1)).astype(np.int32)), axis=1).astype(np.int32)

    print("Run ",t,"  Stats: ",config.Array_N_agents[-1], int(np.sum(config.EI.T_c)), 
    config.nr_excessbirths[-1], "/", config.nr_excessdeaths[-1], np.sum([ag.pop for ag in config.agents]), "\t Happy: ","%.2f" % config.happyMeans[-1],"\t Fisher:",int(config.FisherAgents),"\t GardenFrac",'%.2f' % (config.GardenFraction[-1]),"\t Time tot: ", '%.2f'  % (time()-start_step))
    return 

def update_single_agent(ag,t):
    ''' for agent `ag': perform update step ''' 
    
    #ag.change_tree_pref(-config.treePref_decrease_per_year)
    #ag.calc_new_tree_pref()
    #ag.calc_tree_need()
    #ag.calc_agri_need()

    previous_treeNr = np.copy(np.sum(config.EI.T_c))

    ##########################
    ## HARVEST AGRICULTRE   ##
    ##########################
    C_F_ci_arr = np.where(config.EI.C_F_c[:,ag.c_i])[0]

    if not ag.fisher and config.EI.AnakenaBeach_ind in C_F_ci_arr:
        if config.FisherAgents < config.MaxFisherAgents:
            ag.fisher = True
            config.FisherAgents+=1
            ag.calc_new_tree_pref()
            ag.calc_F_Req()
            ag.calc_T_Req()
            ag.farming_fill=1

    if not ag.fisher:
        for sites in [config.EI.well_suited_cells, config.EI.eroded_well_suited_cells, config.EI.poorly_suited_cells]:
            if ag.F_Req_i  <= np.sum(ag.F_PI_i):
                ag.farming_fill =  (np.sum(ag.F_PI_i) / ag.F_Req_i).clip(max=1)
                break
            # sites with space for more acres
            potential_sites_inds = C_F_ci_arr[np.where(sites[C_F_ci_arr].astype(int)>config.EI.A_F_c[C_F_ci_arr])] #np.dot(neighbours, config.EI.nr_highqualitysites)        
                
            # Tree densities / Carr Cap  on the triangles with potential for agriculture
            TreeDens_onPotentialAgri = config.EI.T_c[potential_sites_inds] / (config.EI.carrying_cap[potential_sites_inds]).clip(1e-5)
            
            # Space for Agriculture: nr of acres not covered by tree and not occupied by agriculture
            TreelessSpaceForAgr = config.EI.A_c[potential_sites_inds] * config.km2_to_acre *  (1- TreeDens_onPotentialAgri) - config.EI.A_F_c[potential_sites_inds] 
            TreelessSpaceForAgr_int = np.floor(TreelessSpaceForAgr) # Round down.
            
            #TEST: if any(TreelessSpaceForAgr > config.EI.nr_highqualitysites[potential_sites_inds]):
            #    print("ERROR in CALC SPACE FOR AGR ", potential_sites_inds, TreelessSpaceForAgr, config.EI.nr_highqualitysites[potential_sites_inds] )
            #    quit()
            
            # TODO   I COULD COMBINE THIS WITH THE TREE BURNING LATER AND JUST TAKE THE ONES WHERE 0 BURNING REQUIRED.

            # The indices of the triangles of ALL acres, that are unoccupied, tree-less and have potential agriculture . e.g. [1,1,1,4,4,5,8,6,13,13,13]
            treeless_sites_inds = np.array([i  for n,i in enumerate(potential_sites_inds) for k in range(int(TreelessSpaceForAgr_int[n])) ], dtype=int)
            treeless_sites_yields = config.EI.F_pi_c[treeless_sites_inds]
            #bestTreelessSitesFirst_inds = treeless_sites_inds[np.argsort(config.EI.agric_yield[treeless_sites_inds]).astype(int)]
            while len(treeless_sites_inds)>0 and ag.F_Req_i > np.sum(ag.F_PI_i):
                s = np.random.choice(np.where(treeless_sites_yields==np.max(treeless_sites_yields))[0]) #bestTreelessSitesFirst_inds[0]
                ag.A_F_i = np.append(ag.A_F_i, treeless_sites_inds[s]).astype(int)
                ag.F_PI_i = np.append(ag.F_PI_i, treeless_sites_yields[s])
                config.EI.A_F_c[treeless_sites_inds[s]]+=1
                # For burning later...
                if config.EI.A_F_c[treeless_sites_inds[s]] == sites[treeless_sites_inds[s]]:
                    potential_sites_inds = np.delete(potential_sites_inds, np.where(potential_sites_inds==treeless_sites_inds[s])[0])
                treeless_sites_inds = np.delete(treeless_sites_inds,s)
                treeless_sites_yields = np.delete(treeless_sites_yields,s)
            
            if ag.F_Req_i  <= np.sum(ag.F_PI_i):  
                ag.farming_fill = (np.sum(ag.F_PI_i) / ag.F_Req_i).clip(max=1)
                break



            # # Nr of trees that the agent will occupy without burning
            # Nr_NewSites_NoBurning = np.ceil(min(ag.AgriNeed - np.sum(ag.MyAgricYields), len(treeless_sites_inds))).astype(int)

            # NewSites_NoBurning = np.random.choice(treeless_sites_inds, size = Nr_NewSites_NoBurning, replace=False)
            # for s in NewSites_NoBurning:
            #     ag.AgricSites = np.append(ag.AgricSites, s).astype(int)
            #     ag.MyAgricYields = np.append(ag.MyAgricYields, config.EI.agric_yield[s])
            #     #ag.AgricYield += config.EI.agric_yield[s]
            #     config.EI.agriculture[s] +=1
            #     treeless_sites_inds.remove(s)

            #     if config.EI.agriculture[s] == sites[s]:
            #         potential_sites_inds = np.delete(potential_sites_inds, np.where(potential_sites_inds==s)[0])
            #     if ag.AgriNeed  <= np.sum(ag.MyAgricYields):  
            #         agriculture_fill = 1.0
            #         break


            if ag.F_Req_i  > np.sum(ag.F_PI_i) and len(potential_sites_inds)>0:  
                ######### BURN ####################   
                '''
                We have now FreeSpace = (1-N_trees/CarrCap) * Area - AgricAcres <1
                But we want it to be 1.
                Hence (1- (Ntrees-DelN)/CarrCap) * Area - AgricAcres >= 1
                then DelN = N - C*(1- (1+AgricAcres)/A)  need to be cut for the next acre
                '''
                # update Space: (should not have anything >=1)
                # TODO BETTER: INSTEAD OF DOING IT AGAIN REMOVE THE CORRESPONDING ENTRIES IN THE UPPER CALCULATION
                #potential_sites_inds = agric_neighbours_inds[np.where(config.EI.sites[agric_neighbours_inds].astype(int)>config.EI.agriculture[agric_neighbours_inds])] #np.dot(neighbours, config.EI.nr_highqualitysites)        
                #if len(potential_sites_inds)==0:
                #    agriculture_fill= len(ag.AgricSites)/ag.AgriNeed
                #else:
                    #TreeDens_onPotentialAgri = config.EI.tree_density[potential_sites_inds] / (config.EI.carrying_cap[potential_sites_inds]).clip(1e-5)
                    #SpaceForAgr = config.EI.EI_triangles_areas[potential_sites_inds] * config.km2_to_acre * (1- TreeDens_onPotentialAgri)  - config.EI.agriculture[potential_sites_inds] 
                
                if any(config.EI.T_c<0):
                    print("Error already here", np.where(config.EI.T_c<0))
                    quit()
                
                # for _ in range(ag.AgriNeed - len(ag.AgricSites)):
                while ag.F_Req_i>np.sum(ag.F_PI_i)  and len(potential_sites_inds)>0:
                    howmanytreesneedtoburn = np.ceil( 
                        config.EI.T_c[potential_sites_inds] - 
                        (1 - (1+config.EI.A_F_c[potential_sites_inds])/(config.EI.A_c[potential_sites_inds]*
                        config.km2_to_acre)) *
                        config.EI.carrying_cap[potential_sites_inds]
                    ).astype(int)
                    # Example: In triangle (2.5 Acres) there are 18 of initial 20 trees)
                    # SpaceForAgr = (1 - 18/20) * 2.5  - 0 = 0.1*2.5 = 0.25
                    # There's only space for a quarter acre.
                    # We need more space--> Burn Trees
                    # DelTree = N - C*(1- (1+AgricAcres)/A) = 18 - 20 * (1 - (1+0)/2.5) = 6.0 
                    # CUT 6 trees, then we have 12 (SpaceForAgr = 1-12/20)*2.5-0 = 1 DONE
                    # NEXT ROUND: 
                    # Space (1-12/20)*2.5 -1 = 0.0 --> DelTree = 12-20*(1-2/2.5)
                    # Would need max of  6 Trees! Then (1-6/20) *1.5 = 1.0xx
                    # HowmanyTreesNeedtoburn =  18 - (0+1- 1/area) * 20 =  11.3 --> 12  
                    # or 18- (1+1-1/area)*20 = 
                    # 18 - 20* (1-(0+1)/1.5) = 

                    if any(howmanytreesneedtoburn<0):
                        print("BURN NEGATIVE TREES!")
                        quit()
                
                    #how_many_to_burn = 1 #  , len(howmanytreesneedtoburn))
                    #cells_to_burn = np.argpartition(howmanytreesneedtoburn, how_many_to_burn-1)[:how_many_to_burn]
                    

                    # Don't burn all trees on one cell. 
                    sites_NotAllTreesBurned = np.where(config.EI.T_c[potential_sites_inds]>howmanytreesneedtoburn)[0]
                    if len(sites_NotAllTreesBurned)>0:
                        chosenCell = np.argmin(howmanytreesneedtoburn[sites_NotAllTreesBurned])
                    else:
                        chosenCell = np.argmin(howmanytreesneedtoburn)
                    #for newSite in ce  Iiiiiiiiiills_to_burn:
                    Site_ind = potential_sites_inds[chosenCell]
                    
                    if howmanytreesneedtoburn[chosenCell]> config.EI.T_c[Site_ind]:
                        print("ERROR: Burn more trees", howmanytreesneedtoburn[chosenCell], " than avail: ",config.EI.T_c[Site_ind] )
                        #print(config.EI.agriculture[Site_ind], (config.EI.EI_triangles_areas[Site_ind]*
                        #config.km2_to_acre), config.EI.nr_highqualitysites[Site_ind].astype(int),
                        #config.EI.carrying_cap[Site_ind])
                    config.fire.append([Site_ind,howmanytreesneedtoburn[chosenCell], t])
                    config.EI.T_c[Site_ind] -= howmanytreesneedtoburn[chosenCell]
                    if config.EI.T_c[Site_ind]<0:
                        print("ERROR!!!! in ",Site_ind)
                        quit()
                    ag.A_F_i= np.append(ag.A_F_i,Site_ind).astype(int)
                    ag.F_PI_i = np.append(ag.F_PI_i, config.EI.F_pi_c[Site_ind])
                    config.EI.A_F_c[Site_ind]+=1

                    if config.EI.A_F_c[Site_ind] == sites[Site_ind]:
                        # Triangle is full, not even burning makes more space for agriculture.
                        #print("Triangle ", Site_ind," is full: ",config.EI.agriculture[Site_ind])
                        potential_sites_inds = np.delete(potential_sites_inds, chosenCell)
                        #if len(potential_sites_inds)==0:
                        #    break
            ag.farming_fill =  (np.sum(ag.F_PI_i) / ag.F_Req_i).clip(max=1)
                    #           burn the area required on the cell with the min of trees too much
                    #           store that we had a fire, store new site (and remove from list) and agent happy





    # NOW FOREST HARVEST

    tree_neighbours_inds = np.where(config.EI.C_T_c[:,ag.c_i])[0]
    #Tree_neighbour_01arr = np.zeros(config.EI.N_els).astype(int)
    #Tree_neighbour_01arr[tree_neighbours_inds] =1 # N_els, 1 for triangles within Res_rad, 0 else


    tree_Nrs = config.EI.T_c[tree_neighbours_inds]
    total_tree_number_inRad = np.sum(tree_Nrs)
    #total_tree_number_inRad = np.dot(config.EI.tree_density, tree_neighbours_inds)
    # OPTION 2
    #tree_arr = [ind for n,ind in enumerate(tree_neighbours_inds) for _ in range(tree_Nrs[n])]
    #cuts = np.random.choice(tree_arr, size=min(int(ag.tree_need),total_tree_number_inRad ), replace=False)
    #config.EI.tree_density[cuts]-=1
    # OPTION 1
    TreeHarvest = 0
    for _ in range(int(ag.T_Req_i)):
        if total_tree_number_inRad==0:
            break 
        #Why according to p? Because if there's no trees left there, this would cause an error.
        nb_tri_ind = np.random.choice(np.where(tree_Nrs>0)[0])#, p=(tree_Nrs>0)/np.sum(tree_Nrs>0))
        cut_tri_ind = tree_neighbours_inds[nb_tri_ind]
        TreeHarvest+=1
        config.EI.T_c[cut_tri_ind] -= 1
        total_tree_number_inRad -=1
        
        tree_Nrs[nb_tri_ind] -=1
        #if tree_Nrs[nb_tri_ind]==0:
        #    tree_Nrs = np.delete(tree_Nrs,nb_tri_ind)
        #    tree_neighbours_inds = np.delete(tree_neighbours_inds, nb_tri_ind)
        
    if any(tree_Nrs<0):
        print("ERROR Trees <0")
        quit()

    ag.tree_fill = (TreeHarvest/ag.T_Req_i).clip(0.,1.)
    #ag.farming_fill


    if ag == config.agents[0]:
        print("Agent ",ag.index, "(pop",ag.pop,"): Pref ", '%.4f' % ag.T_Pref_i,", TreeNeed ", ag.T_Req_i, ", AgriSite/Need ", np.sum(ag.F_PI_i),"/",len(ag.A_F_i), "/",ag.F_Req_i, " previous happy:",ag.h_i, "Tree/agricFill", "%.2f" % ag.tree_fill, "/","%.2f" % ag.farming_fill, ", penalty","%.2f" % ag.penalty )


    h_i_new = np.min([ag.tree_fill, ag.farming_fill])
    if h_i_new < ag.h_i:
        ag.H_i= np.mean([h_i_new, ag.h_i])
    else:
        ag.H_i = h_i_new
    ag.h_i = h_i_new 

    survived = ag.population_change(ag.H_i,t) # REprod and Pop Shock!
    if survived:
        if ag.H_i<config.H_equ:
            move(ag,t)
            config.move+=1
        else:
            ag.penalty, _, _, _, _ ,_= calc_penalty(ag, np.array([ag.c_i]), t) 
            #  P_tot, ag.alpha_T_prime,  ag.alpha_F_prime, [P_W, P_G, P_T, P_F, P_D], NecessaryCond_T, NecessaryCond_F
    # ### 2. Perhaps Reproduce
    # if ag.happy==1.0:
    #     ag.Reproduce(t)

    if len(config.agents)>0 and ag == config.agents[0]:
        print("Agent ",ag.index, ", new pop: ",ag.pop," happy:",ag.h_i)

    ag.calc_new_tree_pref()
    ag.calc_F_Req()
    ag.calc_T_Req()

    if np.any(config.EI.A_F_c<0):
            print("ERROR, Update after reproduce, agric <0", np.where(config.EI.A_F_c<0) )
            quit()

    if not np.sum([len(ag.A_F_i) for ag in config.agents]) == np.sum(config.EI.A_F_c):
        print("ERROR: agricultures do not match")
        quit()

    if previous_treeNr < np.sum(config.EI.T_c):
        print("Agent ", ag.index, " added trees: ", ag.T_Req_i)
        quit()
    return 
        
