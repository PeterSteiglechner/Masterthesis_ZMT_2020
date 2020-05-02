
import numpy as np
from time import time
from copy import copy
import config


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
    config.nr_agents.append(len(config.agents))
    config.nr_trees.append(np.sum(config.EI.tree_density))
    config.nr_pop.append(np.sum([ag.pop for ag in config.agents]))
    config.nr_deaths.append(config.deaths)
    config.deaths = 0
    # config.fires (updated during update)
    #config.agents_stages.append(np.histogram([ag.stage for ag in config.agents], bins=np.arange(config.Nr_AgricStages+1))[0])
    config.agents_stages.append(np.histogram([ag.treePref for ag in config.agents], bins=np.linspace(config.MinTreeNeed, config.init_TreePreference, num=10))[0])

    posSize = [np.array([int(ag.index), ag.x, ag.y, ag.pop]) for ag in config.agents]
    config.agents_pos_size.append(np.array(posSize))

    # ENVIRONMENT:
    config.nr_agricultureSites = np.append(config.nr_agricultureSites, np.sum(config.EI.agriculture))
    if not len(config.agents)==0:
        config.nr_happyAgents = np.append(config.nr_happyAgents, (np.mean([ag.happy for ag in config.agents])))
        config.Penalty_mean = np.append(config.Penalty_mean, (np.mean([ag.penalty for ag in config.agents])) )
        config.Penalty_std = np.append(config.Penalty_std, (np.std([ag.penalty for ag in config.agents])))
        config.NrFisherAgents = np.append(config.NrFisherAgents, config.FisherAgents)
        config.GardenFraction = np.append(config.GardenFraction, np.mean([np.sum(ag.MyAgricYields==config.UpperLandSoilQuality)/max(len(ag.AgricSites),1) for ag in config.agents]))
    else:
        config.nr_happyAgents = np.append(config.nr_happyAgents,0)
        config.Penalty_std = np.append(config.Penalty_std,0)
        config.Penalty_mean = np.append(config.Penalty_std,0)
        config.NrFisherAgents = np.append(config.NrFisherAgents, 0)
        config.GardenFraction =  np.append(config.GardenFraction, 0)
    config.moves = np.append(config.moves, config.move)
    config.moves=0

    config.Fraction_eroded = np.append(config.Fraction_eroded, np.sum(config.EI.agric_yield==config.UpperLandSoilQuality) /(np.sum(config.EI.agric_yield==1)+np.sum(config.EI.agric_yield==config.UpperLandSoilQuality)))
    config.Array_tree_density = np.concatenate((config.Array_tree_density, config.EI.tree_density.reshape((config.EI.N_els,1)).astype(np.int32)), axis=1) 
    config.Array_agriculture = np.concatenate((config.Array_agriculture, config.EI.agriculture.reshape((config.EI.N_els,1)).astype(np.int32)), axis=1) 
    config.Array_populationOccupancy = np.concatenate((config.Array_populationOccupancy, config.EI.populationOccupancy.reshape((config.EI.N_els,1)).astype(np.int32)), axis=1) 

    print("Run ",t,"  Stats: ",config.nr_agents[-1], int(config.nr_trees[-1]), 
    config.nr_deaths[-1], config.nr_pop[-1], "\t Happy: ","%.2f" % config.nr_happyAgents[-1],"\t Fisher:",int(config.FisherAgents),"\t GardenFrac",'%.2f' % (config.GardenFraction[-1]),"\t Time tot: ", '%.2f'  % (time()-start_step))


    return 

def update_single_agent(ag,t):
    ''' for agent `ag': perform update step ''' 
    
    #ag.change_tree_pref(-config.treePref_decrease_per_year)
    ag.calc_new_tree_pref()
    ag.calc_tree_need()
    ag.calc_agri_need()

    previous_treeNr = np.copy(np.sum(config.EI.tree_density))

    ##########################
    ## HARVEST AGRICULTRE   ##
    ##########################


    agric_neighbours_inds = np.array(config.EI.AgricNeighbours_of_triangles[ag.triangle_ind])

    if not ag.fisher and config.EI.AnakenaBeach_ind in agric_neighbours_inds:
        if config.FisherAgents < config.MaxFisherAgents:
            ag.fisher = True
            config.FisherAgents+=1
            ag.calc_agri_need()
            ag.calc_tree_need()
    
    if ag.AgriNeed  <= np.sum(ag.MyAgricYields):
        # agent has enough agriculture.
        agriculture_fill = 1.0
    else:  # NEED MORE AGRICULTURE
        # get potential sites (non-occupied and allowed)

        for sites in [config.EI.nr_highqualitysites, config.EI.nr_lowqualitysites]:
            if ag.AgriNeed  <= np.sum(ag.MyAgricYields):
                #agriculture_fill =  np.sum(config.EI.agric_yield[ag.AgricSites]) / ag.AgriNeed
                break
            potential_sites_inds = agric_neighbours_inds[np.where(sites[agric_neighbours_inds].astype(int)>config.EI.agriculture[agric_neighbours_inds])] #np.dot(neighbours, config.EI.nr_highqualitysites)        
                
            # Tree densities / Carr Cap  on the triangles with potential for agriculture
            TreeDens_onPotentialAgri = config.EI.tree_density[potential_sites_inds] / (config.EI.carrying_cap[potential_sites_inds]).clip(1e-5)
            
            # Space for Agriculture: nr of acres not covered by tree and not occupied by agriculture
            TreelessSpaceForAgr = config.EI.EI_triangles_areas[potential_sites_inds] * config.km2_to_acre *  (1- TreeDens_onPotentialAgri) - config.EI.agriculture[potential_sites_inds] 
            TreelessSpaceForAgr_int = np.floor(TreelessSpaceForAgr) # Round down.
            
            #TEST: if any(TreelessSpaceForAgr > config.EI.nr_highqualitysites[potential_sites_inds]):
            #    print("ERROR in CALC SPACE FOR AGR ", potential_sites_inds, TreelessSpaceForAgr, config.EI.nr_highqualitysites[potential_sites_inds] )
            #    quit()
            
            # TODO   I COULD COMBINE THIS WITH THE TREE BURNING LATER AND JUST TAKE THE ONES WHERE 0 BURNING REQUIRED.

            # The indices of the triangles of ALL acres, that are unoccupied, tree-less and have potential agriculture . e.g. [1,1,1,4,4,5,8,6,13,13,13]
            treeless_sites_inds = np.array([i  for n,i in enumerate(potential_sites_inds) for k in range(int(TreelessSpaceForAgr_int[n])) ], dtype=int)
            treeless_sites_yields = config.EI.agric_yield[treeless_sites_inds]
            #bestTreelessSitesFirst_inds = treeless_sites_inds[np.argsort(config.EI.agric_yield[treeless_sites_inds]).astype(int)]
            while len(treeless_sites_inds)>0 and ag.AgriNeed > np.sum(ag.MyAgricYields):
                s = np.random.choice(np.where(treeless_sites_yields==np.max(treeless_sites_yields))[0]) #bestTreelessSitesFirst_inds[0]
                ag.AgricSites = np.append(ag.AgricSites, treeless_sites_inds[s]).astype(int)
                ag.MyAgricYields = np.append(ag.MyAgricYields, treeless_sites_yields[s])
                config.EI.agriculture[treeless_sites_inds[s]]+=1
                # For burning later...
                if config.EI.agriculture[treeless_sites_inds[s]] == sites[treeless_sites_inds[s]]:
                    potential_sites_inds = np.delete(potential_sites_inds, np.where(potential_sites_inds==treeless_sites_inds[s])[0])
                treeless_sites_inds = np.delete(treeless_sites_inds,s)
                treeless_sites_yields = np.delete(treeless_sites_yields,s)
            
            if ag.AgriNeed  <= np.sum(ag.MyAgricYields):  
                agriculture_fill = 1.0
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


            if ag.AgriNeed  > np.sum(ag.MyAgricYields) and len(potential_sites_inds)>0:  
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
                
                if any(config.EI.tree_density<0):
                    print("Error already here", np.where(config.EI.tree_density<0))
                    quit()
                
                # for _ in range(ag.AgriNeed - len(ag.AgricSites)):
                while ag.AgriNeed>np.sum(ag.MyAgricYields)  and len(potential_sites_inds)>0:
                    howmanytreesneedtoburn = np.ceil( 
                        config.EI.tree_density[potential_sites_inds] - 
                        (1 - (1+config.EI.agriculture[potential_sites_inds])/(config.EI.EI_triangles_areas[potential_sites_inds]*
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
                    sites_NotAllTreesBurned = np.where(config.EI.tree_density[potential_sites_inds]>howmanytreesneedtoburn)[0]
                    if len(sites_NotAllTreesBurned)>0:
                        chosenCell = np.argmin(howmanytreesneedtoburn[sites_NotAllTreesBurned])
                    else:
                        chosenCell = np.argmin(howmanytreesneedtoburn)
                    #for newSite in ce  Iiiiiiiiiills_to_burn:
                    Site_ind = potential_sites_inds[chosenCell]
                    
                    if howmanytreesneedtoburn[chosenCell]> config.EI.tree_density[Site_ind]:
                        print("ERROR: Burn more trees", howmanytreesneedtoburn[chosenCell], " than avail: ",config.EI.tree_density[Site_ind] )
                        #print(config.EI.agriculture[Site_ind], (config.EI.EI_triangles_areas[Site_ind]*
                        #config.km2_to_acre), config.EI.nr_highqualitysites[Site_ind].astype(int),
                        #config.EI.carrying_cap[Site_ind])
                    config.fire.append([Site_ind,howmanytreesneedtoburn[chosenCell], t])
                    config.EI.tree_density[Site_ind] -= howmanytreesneedtoburn[chosenCell]
                    if config.EI.tree_density[Site_ind]<0:
                        print("ERROR!!!! in ",Site_ind)
                        quit()
                    ag.AgricSites= np.append(ag.AgricSites,Site_ind).astype(int)
                    ag.MyAgricYields = np.append(ag.MyAgricYields, config.EI.agric_yield[Site_ind])
                    config.EI.agriculture[Site_ind]+=1

                    if config.EI.agriculture[Site_ind] == sites[Site_ind]:
                        # Triangle is full, not even burning makes more space for agriculture.
                        #print("Triangle ", Site_ind," is full: ",config.EI.agriculture[Site_ind])
                        potential_sites_inds = np.delete(potential_sites_inds, chosenCell)
                        #if len(potential_sites_inds)==0:
                        #    break
            agriculture_fill =  np.sum(ag.MyAgricYields) / ag.AgriNeed
                    #           burn the area required on the cell with the min of trees too much
                    #           store that we had a fire, store new site (and remove from list) and agent happy





    # NOW FOREST HARVEST

    tree_neighbours_inds = np.array(config.EI.TreeNeighbours_of_triangles[ag.triangle_ind])
    #Tree_neighbour_01arr = np.zeros(config.EI.N_els).astype(int)
    #Tree_neighbour_01arr[tree_neighbours_inds] =1 # N_els, 1 for triangles within Res_rad, 0 else


    tree_Nrs = config.EI.tree_density[tree_neighbours_inds]

    total_tree_number_inRad = np.sum(tree_Nrs)

    # OPTION 2
    #tree_arr = [ind for n,ind in enumerate(tree_neighbours_inds) for _ in range(tree_Nrs[n])]
    #cuts = np.random.choice(tree_arr, size=min(int(ag.tree_need),total_tree_number_inRad ), replace=False)
    #config.EI.tree_density[cuts]-=1
    # OPTION 1
    for _ in range(min(int(ag.tree_need),total_tree_number_inRad )):
        #Why according to p? Because if there's no trees left there, this would cause an error.
        cut_tri_ind = np.random.choice(tree_neighbours_inds, p=(tree_Nrs>0)/np.sum((tree_Nrs>0)))
        config.EI.tree_density[cut_tri_ind] -= 1
        tree_Nrs[np.where(tree_neighbours_inds == cut_tri_ind)[0][0]] -=1
        
    if any(tree_Nrs<0):
        print("ERROR Trees <0")
        quit()

    tree_fill = (total_tree_number_inRad/int(ag.tree_need)).clip(0.,1.)

    ag.tree_fill = tree_fill
    ag.agriculture_fill = agriculture_fill


    if ag == config.agents[0]:
        print("Agent ",ag.index, "(pop",ag.pop,"): Pref ", '%.4f' % ag.treePref,", TreeNeed ", ag.tree_need, ", AgriSite/Need ", np.sum(ag.MyAgricYields),"/",len(ag.AgricSites), "/",ag.AgriNeed, " previous happy:",ag.happy, "Tree/agricFill", "%.2f" % ag.tree_fill, "/","%.2f" % ag.agriculture_fill )


    if tree_fill>=1 and agriculture_fill>=1:
        ag.happy=1.0
        #ag.happyPeriod+=1

        ###########################
        ###   CHECK PENALTY    ####
        ###########################
        # ag.mv_inds exists
        config.EI.agriculture[ag.AgricSites] -=1
        if ag.fisher: config.FisherAgents-=1

        WhichTrianglesToCalc_inds = [ag.triangle_ind]
        total_penalties,  _, penalties, _= ag.calc_penalty(WhichTrianglesToCalc_inds, t)

        config.EI.agriculture[ag.AgricSites] +=1
        if ag.fisher: config.FisherAgents+=1

        # total_penalties, maske, penalties, masken
        if any([p[0]==1.0 for p in penalties]): #ag.MaxToleratedPenalty:
            if ag == config.agents[0]:
                print("First agent moved: ", penalties)
            ag.move_water_agric(t)
            
        else:
            ag.penalty = total_penalties


    else:
        #if tree_fill<1:
        #    ag.treePref = (ag.treePref- config.treePref_change_per_BadYear)# * config.dStage) # half stage down
        #if agriculture_fill<1:
        #    ag.treePref = (ag.treePref + config.treePref_change_per_BadYear)
        #    #ag.change_tree_pref(config.treePref_change_per_BadYear)# * config.dStage)
        #ag.treePref = ag.treePref.clip(config.MinTreeNeed, config.init_TreePreference)

        # ag.happy from last time!!
        # CASE 1: Agent was happy before --> no one should die
        # CASE 2: Agent was happy before --> now 0 trees / agric --> mean? or no one
        # CASE 3: Agent was slightly unhappy before, moved, still slightly unhappy: --> current unhappy should die (or mean?)
        # CASe 4: Previously 0, now 0.9 happy: only 10% should die this time.
        
        #ag.happy= max(ag.happy, np.mean([tree_fill, agriculture_fill]))   # Max of last happy or mean of this year's harvest
        
        ag.happy=np.mean([np.min([tree_fill, agriculture_fill]), ag.happy])
        survived =  ag.pop_shock() # survived
        if survived and ag.happy<0.8:
            ag.move_water_agric(t)
            ag.happy =  np.min([tree_fill, agriculture_fill])  


    ### 2. Perhaps Reproduce
    if ag.happy==1.0:
        ag.Reproduce(t)

    if ag == config.agents[0]:
        print("Agent ",ag.index, ", new pop: ",ag.pop," happy:",ag.happy)

    
    ag.calc_new_tree_pref()
    ag.calc_tree_need()
    ag.calc_agri_need()

    if np.any(config.EI.agriculture<0):
            print("ERROR, Update after reproduce, agric <0", np.where(config.EI.agriculture<0) )
            quit()

    if not np.sum([len(ag.AgricSites) for ag in config.agents]) == np.sum(config.EI.agriculture):
        print("ERROR: agricultures do not match")

    if previous_treeNr < np.sum(config.EI.tree_density):
        print("Agent ", ag.index, " added trees: ", ag.tree_need)
    return 
        
