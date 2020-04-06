#!/usr/bin/env python

# Peter Steiglechner, contains the classes agents

import config 

import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from time import time

from observefunc import plot_movingProb

class agent:
    ''' Agent of Base HopVes'''
    def __init__(self, x, y, params): #resource_search_radius, moving_radius, fertility):
        ''' location x,y. 
        Agent_type = "individual"/"household", 
        map_type="EI"/"unitsquare", 
        params = {'fertility_rate':...}
        '''
        self.index = config.index_count
        config.index_count += 1
        # Position on map
        self.x = x 
        self.y = y
        
        self.triangle_ind = -1
        self.triangle_midpoint = [-1,-1]
        self.mv_inds = []
        
        # Search for resource in area with radius 
        self.tree_search_radius = params['tree_search_radius']
        self.agriculture_radius = params['agriculture_radius']
        # If not enough resources found, move to random new position in area moving_radius
        self.moving_radius = params['moving_radius']
 
        self.reprod_rate = params["reproduction_rate"]
        
        self.pop = config.init_pop

        self.stage = 0  # of config.Nr_AgricStages
        # 0 --> 100% Tree Need, 
        # Nr_AgricStages --> (100%-MinTreeNeed) Tree and Agri= Rest Agriculture
        

        # calculate total Tree and Agriculture Need:
        self.AgriNeed = 0
        self.tree_need = 0
        self.treePref = config.init_TreePreference
        self.calc_tree_need()
        self.calc_agri_need()

        self.AgricSites = []

        self.happy=True
        #self.happyPeriod=0

        self.alpha_w = config.alpha_w#0.3
        self.alpha_t = config.alpha_t #0.2
        self.alpha_p = config.alpha_p#0.3
        self.alpha_a = config.alpha_a# 0.2
        self.alpha_m = config.alpha_m

        return 
    
    def calc_tree_need(self):
        self.tree_need = int(config.tree_need_per_capita*self.pop * (1-config.dStage * self.stage))
        return 
    def calc_agri_need(self):
        self.AgriNeed = np.round(self.pop * config.agricSites_need_per_Capita *  self.stage / config.Nr_AgricStages).astype(int)

    def change_tree_pref(self, amount):
        self.treePref = np.clip(self.treePref + amount, config.MinTreeNeed, 1)  # e.g. -config.treePref_decrease_per_year
        #self.stage = (1-self.treePref)  - ((1-self.treePref)  %  config.dStage) #
        self.stage = int((1-self.treePref) / config.dStage)
        self.calc_tree_need()
        self.calc_agri_need()
        return 


    # Helper Function
    def euclid_distance_1vec1ag(self, m,ag):
        return np.linalg.norm(m-np.array([ag.x, ag.y]))


    def pop_shock(self):
        # REMOVE SOME POPULATION
        #dead_pop = np.random.randint(0,self.pop/2) # TODO: Make a decision if fraction dies or fixed number! #min(config.HowManyDieInPopulationShock, self.pop)
        dead_pop = np.floor(config.FractionDeathsInPopShock * self.pop).astype(int) 
        config.deaths += dead_pop
        self.pop -= dead_pop
        config.EI.populationOccupancy[self.triangle_ind]-=dead_pop

        # CHECK IF AGENT IS STILL LARGE ENOUGH OR DIES
        if self.pop < config.init_pop:
            config.deaths+= self.pop
            config.EI.agentOccupancy -=1
            config.EI.populationOccupancy[self.triangle_ind] -= dead_pop
            for i in self.AgricSites:
                config.EI.agriculture[i]-=1
            self.AgricSites=[]
            config.agents.remove(self)
            print("Agent ",self.index," died.")
            return False # i.e. not survived
        
        # REDUCE AGRICULTURE SITES
        self.calc_agri_need()
        reduceAgric = int(len(self.AgricSites)- self.AgriNeed)#self.AgriNeed*(self.pop-config.init_pop)/self.pop)
        for site in np.arange(reduceAgric)[::-1]: # need the [::-1] s.t. i remove large indices at first. E.g. [0,1,2] remove 3 elements: remove 2, 1 then 0 works, but remove 0 then 1, then 2 does not work.
            config.EI.agriculture[self.AgricSites[site]]-=1
            self.AgricSites.remove(self.AgricSites[site])
        
        return True # i.e. survived



    def move_water_agric(self, t):
       
        # CLEAR YOUR OLD SPACE!
        config.EI.agentOccupancy[self.triangle_ind] -=1
        config.EI.populationOccupancy[self.triangle_ind] -=self.pop
        for site in self.AgricSites:
            config.EI.agriculture[site] -=1
        self.AgricSites=[]

        # GET POSSIBLE TRIANGLES!
        # get triangles within moving radius, # values are indices of EI.EI_triangles etc.
        self.mv_inds = [n for n,m in enumerate(config.EI.EI_midpoints) 
                    if self.euclid_distance_1vec1ag(m, self)<self.moving_radius]
       
        # Loop through mv_inds and assign tree densitiy within resource search radius 
        # mvTris_inds_arr gives for every column (triangles within mv radius) a 1 in the column if the corresponding triangle (row) is within the radius.
        mvTris_inds_arr_tree = np.zeros([config.EI.N_els, len(self.mv_inds)], dtype=np.uint8)
        mvTris_inds_arr_agric = np.zeros([config.EI.N_els, len(self.mv_inds)], dtype=np.uint8)
        for i,ind in enumerate(self.mv_inds):
            mvTris_inds_arr_tree[config.EI.TreeNeighbours_of_triangles[ind], i]=1
            mvTris_inds_arr_agric[config.EI.AgricNeighbours_of_triangles[ind],i]=1

        
        ############################# 
        ## TREE PENALTY  ############
        #############################
        # Tree penalty is high if the number of trees within resource search radius is small.
        
        # use mvTris_inds_arr (0,1 -array) in the dot product with the tree density: sum(tree_densities at triangles within moving radius)
        reachable_trees = np.dot(config.EI.tree_density, mvTris_inds_arr_tree) 
        
        #OLD: tree_penalty = 1 - (self.treePref-config.MinTreeNeed)/config.MinTreeNeed * tree_densities_aroundSites / (config.BestTreeNr_forNewSpot)
        # if tree_densities_around > config.BestTree -->> Penalty =0 
        # if treePref = 1 and tree_densities_aroundSites < config.BEstTree_forNewSpot: Large Penalty : self.treePref * tree_dens_aroundSites<config.Best * (1-treedensAround/Best)
        tree_penalty = self.treePref * (1-reachable_trees/config.BestTreeNr_forNewSpot).clip(0)
        survivalCond_tree = (reachable_trees>self.tree_need).astype(int)
        tree_penalty[np.where(survivalCond_tree==False)[0]]=1

        #min_tree_penalty = 0 if maximumDensityTreesNearby
        #max_tree_penalty = 1 # if trees = 0 nearby
        
        
        ############################# 
        ## Pop Densi PENALTY  #######
        #############################               
        # Pop_density at NearBy = sum(pop_densitie at triangles within moving rad) --> ppl/km^2
        pop_density_atnb=np.dot(config.EI.populationOccupancy, mvTris_inds_arr_agric) / (self.agriculture_radius**2 * np.pi) #* 1e6
        pop_density_penalty = pop_density_atnb / config.MaxPopulationDensity
        #if any(pop_density_penalty>1):
        #    print("There's an area with too much population!!")

        # Additionally to a penalty. Triangles that already exceed the maximum Population Density are excluded!!
        population_cond = pop_density_penalty < 1  # smalle than Max!

        ############################# 
        ## Water     PENALTY  #######
        #############################
        # Water Penalty of all triangles in moving radius       
        water_penalty =config.EI.water_penalties[self.mv_inds]
        

        ############################# 
        ## Agriculture PENALTY  #####
        #############################
        # Agriculture Sites: The ones where tree_dens < MaxTreeDensForAgri   and   potentialSite
        #PotentialAgriSites_ind = config.EI.nr_highqualitysites  # note: TODO remove the highquality site from nr_highqualitysites after selected by agent
        # tree densities in all Cells!!  --> N_els shape Bool Array.
        
        # NEW
        '''
        OPTION 1: TAKE THE REALlY AVAILABLE (no other agent, and no trees blocking) and then caluclate the Penalty!
        potential_sites_inds = np.array(self.mv_inds)[np.where(config.EI.nr_highqualitysites[self.mv_inds].astype(int)>config.EI.agriculture[self.mv_inds])] #np.dot(neighbours, config.EI.nr_highqualitysites)        
        TreeDens_onPotentialAgri = config.EI.tree_density[potential_sites_inds] / (config.EI.carrying_cap[potential_sites_inds]).clip(1e-5)
        SpaceForAgr = (np.floor((1-TreeDens_onPotentialAgri) * config.EI.EI_triangles_areas[potential_sites_inds] * config.km2_to_acre)).astype(int) - config.EI.agriculture[potential_sites_inds]
        if any(SpaceForAgr<0):
            print("ERROR, space for agr negative in Move!")
        SpaceForAgr_allcells=np.zeros([config.EI.N_els], dtype=np.uint8)
        SpaceForAgr_allcells[potential_sites_inds]=SpaceForAgr
        nrAvailAgrisSites_aroundSites =  np.dot(SpaceForAgr_allcells, mvTris_inds_arr)       
        
        agriculture_penalty = ((1-self.treePref) * (self.AgriNeed - nrAvailAgrisSites_aroundSites)).clip(0)#, a_min= 0, a_max = 1000)
        max_agriculture_penalty = (1-config.MinTreeNeed) * (config.max_pop_per_household * config.agricSites_need_per_Capita - 0)
        agriculture_penalty = agriculture_penalty/max_agriculture_penalty
        '''
        # OPTION 2: Take all non-taken agric sites (perhaps with trees). 
        nrAvailAgrisSites_aroundSites = np.dot(config.EI.nr_highqualitysites-config.EI.agriculture, mvTris_inds_arr_agric)
        maxNeededAgric = np.ceil(config.max_pop_per_household*config.agricSites_need_per_Capita)
        agriculture_penalty = (1-self.treePref)/config.MinTreeNeed * (maxNeededAgric - nrAvailAgrisSites_aroundSites.clip(min=0, max=maxNeededAgric))/maxNeededAgric
        
        survivalCond_agric =  nrAvailAgrisSites_aroundSites>self.AgriNeed
        
        ################################
        #####     MAP PENALTY   ########
        ################################

        map_penalty =  abs(1./ (max(config.EI.EI_midpoints_elev)-config.SweetPointSettlementElev) * (config.EI.EI_midpoints_elev[self.mv_inds] - config.SweetPointSettlementElev))
        map_penalty = map_penalty  

        
        ################################
        ## Allowed Settlements! #####
        ################################
        slopes_cond = config.EI.EI_midpoints_slope[self.mv_inds] < config.MaxSettlementSlope
        # AND POPULATION DENSITY

       
       ############# ALPHAS #################
        # SUM OF ALPHA Should BE 1 then Max and Min Prob are 1 and 0!
        #alpha_w= 0.3
        #alpha_t = 0.2
        #alpha_p = 0.2
        #alpha_a = 0.2
        #alpha_m = 0.1

        maske = np.array(slopes_cond)*np.array(population_cond) 
        maske_plot = 2*survivalCond_agric +4*survivalCond_tree  + 1*maske #map_penalty
        maske = maske * survivalCond_agric * survivalCond_tree

        #_ = (self.alpha_w * water_penalty,   
        #     self.alpha_t*tree_penalty,
        #     self.alpha_p * pop_density_penalty,
        #     self.alpha_a * agriculture_penalty,
        #     self.alpha_m * map_penalty,
        #     maske)

        total_penalties = ( self.alpha_w * water_penalty +  self.alpha_t* tree_penalty + self.alpha_p * pop_density_penalty + self.alpha_a * agriculture_penalty + self.alpha_m * map_penalty) 
        
        probabilities= maske * np.exp(-total_penalties)#(1-total_penalties))

        ####################
        ##   MOVE   #######
        ####################
        if any(probabilities>0):
            probabilities=probabilities/np.sum(probabilities)

            # OPTION 1 choose one random of the 5 lowest penalties
            # OPtion 2 choose according to p.
            # self.triangle_ind = self.mv_inds[np.random.choice(np.argpartition(probabilities, -config.NrOfEquivalentBestSitesToMove)[-config.NrOfEquivalentBestSitesToMove:])]
            #self.triangle_ind = self.mv_inds[np.argmax(probabilities)]
            self.triangle_ind = np.random.choice(self.mv_inds, p= probabilities)
        else:
            #print("Agent", self.index, " can't find allowed space. All probs ==0.")
            self.triangle_ind = np.random.choice(self.mv_inds)

        self.triangle_midpoint = config.EI.EI_midpoints[self.triangle_ind]


        ###########################################
        ####   PLOT PENALTIES FOR A FEW AGENTS   ##
        ###########################################
        if config.analysisOn==True and self.index<5:
            print("Saving Agent ", self.index,"'s penalty")
            config.AG0_mv_inds=np.array(self.mv_inds)
            config.AG0_total_penalties = total_penalties * maske
            config.AG0_water_penalties =  water_penalty
            config.AG0_tree_penalty =  tree_penalty
            config.AG0_pop_density_penalty =  pop_density_penalty
            config.AG0_agriculture_penalty =  agriculture_penalty
            config.AG0_map_penalty = map_penalty
            config.AG0_nosettlement_zones = maske_plot
            config.AG0_MovingProb =probabilities 

            plot_movingProb(t,self,self.triangle_midpoint)
            plt.close()


        # TODO!!! NEED TO USE BARYcEnTRIC CORRD TO GET NON_MIDPOINT x,y
        self.x = self.triangle_midpoint[0]
        self.y = self.triangle_midpoint[1] 
        config.EI.agentOccupancy[self.triangle_ind] +=1
        config.EI.populationOccupancy[self.triangle_ind] += self.pop
            
        return 

    def Reproduce(self, t):
        # EACH PERSON HAS A reprod_rate Probability of copying themselves.
        randnrs = np.random.random(int(self.pop))
        newborns = np.sum(randnrs<self.reprod_rate)
        self.pop+=newborns
        config.EI.populationOccupancy[self.triangle_ind] +=newborns

        # IF Population TOO LARGE; SPlit AGENT
        if self.pop>config.max_pop_per_household:
            child = copy(self)
            child.index = config.index_count
            config.index_count+=1
            config.EI.agentOccupancy[child.triangle_ind]+=1
            child.pop = int(config.init_pop)
            child.calc_agri_need()
            child.calc_tree_need()
            child.AgricSites=[] # Note, if child doesn't have any garden, we also do not clear space when moving
            child.move_water_agric(t)

            # Adjust parent agent and reduce its agriculture.
            self.pop -= int(config.init_pop)
            self.calc_tree_need()
            self.calc_agri_need()
            config.agents.append(child)
            reduceAgric = int(len(self.AgricSites)- self.AgriNeed)#self.AgriNeed*(self.pop-config.init_pop)/self.pop)
            for site in np.arange(reduceAgric)[::-1]: # need the [::-1] s.t. i remove large indices at first. E.g. [0,1,2] remove 3 elements: remove 2, 1 then 0 works, but remove 0 then 1, then 2 does not work.
                config.EI.agriculture[self.AgricSites[site]]-=1
                self.AgricSites.remove(self.AgricSites[site])



def init_agents():#N_agents, init_option, agent_type, map_type):
    config.agents=[]
    for _ in range(config.N_agents):
        counter = 0
        searching_for_new_spot=True
        while searching_for_new_spot:        
            # init_option = "anakena"
            midp = config.EI.transform([520,config.EI.pixel_dim[0]-480]) # This is roughly the coast anakena where they landed.
            w = 0.1 # Random number (in km)
            x = np.random.random()*w - w/2 + midp[0]
            y = np.random.random()*w - w/2 + midp[1] 
            ind, _, m_meter = config.EI.get_triangle_of_point([x,y], config.EI_triObject)
            if not ind==-1:
                searching_for_new_spot = False
            if counter>100:
                print("COUNTERBREAK!")
                break
        ag = agent(x, y, config.params)#param['resource_search_radius'], param['moving_radius'], param['fertility'] )  # create agent
        ag.triangle_ind = ind
        ag.triangle_midpoint = m_meter
        config.EI.agentOccupancy[ind]+=1
        config.EI.populationOccupancy[ind] += ag.pop

        config.agents.append(ag)  # add agent to list
    return 

