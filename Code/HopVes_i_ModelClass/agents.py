#!/usr/bin/env python

# Peter Steiglechner, contains the classes agents

import config 

import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from time import time


class agent:
    ''' Agent of Base HopVes'''
    def __init__(self, x, y, params): #resource_search_radius, moving_radius, fertility):
        ''' location x,y. 
        Agent_type = "individual"/"household", 
        map_type="EI"/"unitsquare", 
        params = {'fertility_rate':...}
        '''
        self.index = config.index_count
        config.index_count+=1
        # Position on map
        self.x = x 
        self.y = y
        # Search for resource in area with radius 
        self.resource_search_radius = params['resource_search_radius']
        # If no resource found, move to random new position in area moving_radius
        self.moving_radius = params['moving_radius']
        # Initially agents are not hungry
        self.hungry=False
        
        if config.agent_type == "individual":
            if "fertility" in params:
                # Probability to reproduce
                self.fertility = params["fertility"]
        if config.agent_type == "household":
            if "reproduction_rate" in params:
                self.reprod_rate = params["reproduction_rate"]
                # Initially agents are not hungry
            if "init_pop" in params:
                # Population in household
                self.pop = params["init_pop"]
            if "tree_need_per_capita" in params:
                # nutrition need per capita
                self.tree_need_per_capita = params["tree_need_per_capita"]
            if "max_pop_per_household" in params:
                self.max_pop_per_household = params["max_pop_per_household"]
            # calculate total nutrition need:
            self.tree_need = 0
            self.calc_tree_need()
        if config.map_type == "EI":
            self.triangle_ind = -1
            self.triangle_midpoint = [-1,-1]
        return 
    
    def calc_tree_need(self):
        self.tree_need = self.tree_need_per_capita*self.pop
        return 

    # Helper Function
    def euclid_distance_2ags(self, a, b):
        return ((a.x-b.x)**2 + (a.y-b.y)**2 )**0.5
    def euclid_distance_1vec1ag(self, m,ag):
        return np.linalg.norm(m-np.array([ag.x, ag.y]))

    def search_resource_treeAgent(self):
        ''' returns list of nearby trees around the selected agent '''
        #global trees
        found = [tree for tree in config.trees 
                    if self.euclid_distance_2ags(tree, self) < self.resource_search_radius]
        return len(found), found
    
    def search_resource_treedensity(self):
        #global EI
        ''' returns the sum of nearby trees and the indices of the neighbour cells'''
        nb_inds = [n for n in range(config.EI.N_els) 
                    if self.euclid_distance_1vec1ag(config.EI.EI_midpoints[n], self)<self.resource_search_radius]
        has_trees = [ind for ind in nb_inds if config.EI.tree_density[ind]>0]
        return np.sum(config.EI.tree_density[nb_inds]), has_trees


    def move_randomHop(self):
        ''' assigns a new position to selected agent within moving_radius circle
            (select uniform dx,dy, then check if this is in bounding box and in circle?
        '''
        searching_for_new_spot = True

        while(searching_for_new_spot):
            # Choose dx and dy in square of (moving_rad, moving_rad)
            dx = self.moving_radius* (2* np.random.random() - 1)   # value between: [-moving_rad, moving_rad]
            dy = self.moving_radius* (2* np.random.random() - 1)
            
            if config.map_type=="EI":
                ind_triangle, m, _corner = config.EI.get_triangle_of_point([self.x+dx,self.y+dy])
                Is_in_BoundingShape = (not ind_triangle ==-1)
            elif config.map_type=="unitsquare":
                Is_in_BoundingShape = self.x+dx <1 and self.x+dx >0 and self.y+dy<1 and self.y+dy>0    # check if within the Map
            else:
                config.wrongMapType()
            
            if (
                (dx**2+dy**2)**0.5 < self.moving_radius**2 and       # check if in circle
                Is_in_BoundingShape
            ):    
                # then put agent on this location
                searching_for_new_spot = False     
                self.x += dx
                self.y += dy
                if config.map_type=="EI":
                    self.triangle_midpoint=m
                    self.triangle_ind=ind_triangle
                    config.EI.agentOccupancy[ind_triangle] -=1
        return 
    def report(self):
        report = np.array([self.index, self.x, self.y, self.pop if config.agent_type=="household" else 1])
        return report, ['index', 'x', 'y', 'Pop']

# Init
def init_agents_EI():#N_agents, init_option, agent_type):
    #global agents, params
    #global EI
    config.agents=[]
    for _ in range(config.N_agents):
        counter = 0
        searching_for_new_spot=True
        while searching_for_new_spot:        
            if config.init_option=="equally":
                dx,dy =[a-b for a,b in zip(config.EI.corners['upper_right'],config.EI.corners['lower_left'])]
                corner = config.EI.corners['lower_left']
                x = np.random.random()*dx + corner[0]
                y = np.random.random()*dy + corner[1]
            elif config.init_option=="anakena":
                # init_option = "anakena"
                midp = config.EI.transform([520,config.EI.pixel_dim[0]-480]) # This is roughly the coast anakena where they landed.
                w = 100 # Random number (in m)
                x = np.random.random()*w - w/2 + midp[0]
                y = np.random.random()*w - w/2 + midp[1] 
            ind, _, m_meter = config.EI.get_triangle_of_point([x,y])
            if not ind==-1:
                searching_for_new_spot = False
            if counter>100:
                print("COUNTERBREAK!")
                break
        ag = agent(x, y, config.params)#param['resource_search_radius'], param['moving_radius'], param['fertility'] )  # create agent
        
        #ind, m, _corner = get_triangle_of_point([ag.x, ag.y])
        ag.triangle_ind = ind
        ag.triangle_midpoint = m_meter
        config.EI.agentOccupancy[ind]+=1

        config.agents.append(ag)  # add agent to list
    return 



# Init
def init_agents_UnitSquare():#N_agents, init_option, agent_type):
    #global agents
    #global params
    config.agents=[]
    for _ in range(config.N_agents):
        if config.init_option=="equally":# Equally distributed
            x = np.random.random()   # equally distributed initial agents
            y = np.random.random()   # 
        elif config.init_option=="corner":
            # Agents initialised at a corner
            x = np.random.random()*0.15 + 0.1
            y = 0
        else:
            config.wrongConfigOption()
        ag = agent(x, y, config.params)  # create agent
        config.agents.append(ag)  # add agent to list
    return 


def init_agents():#N_agents, init_option, agent_type, map_type):
    #global agents, params
    if config.agent_type=="household":
        config.params = config.params_hh
    elif config.agent_type=="individual":
        config.params = config.params_individual
    else:
        config.wrongAgentType()
    if config.map_type=="EI":
        init_agents_EI()#config.N_agents, config.init_option, config.agent_type)
    elif config.map_type=="unitsquare":
        init_agents_UnitSquare()#config.N_agents, config.init_option, config.agent_type)
    else: 
        config.wrongMapType()
    return 



###############################
###  FOR unitsquare MODEL   ###
###############################


# Define Trees / resource class
class Tree:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def init_trees():
    config.trees=[]
    for _ in range(int(config.N_trees)):
        x = np.random.random()
        y = np.random.random()
        tree = Tree(x, y)  # create tree
        config.trees.append(tree) # add Tree to list
    return 