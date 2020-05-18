#!/usr/bin/env python

# Peter Steiglechner, contains the classes agents

import config

import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from time import time
import scipy.stats
import pickle

from observefunc import plot_movingProb

from moving import move, calc_penalty

class agent:
    ''' Agent of Base HopVes'''
    def __init__(self, x, y, pop): #resource_search_radius, moving_radius, fertility):
        self.index = config.index_count
        config.index_count += 1
        self.x = x 
        self.y = y
        self.c_i = -1
        self.vec_c_i = [-1,-1]
        self.r_T = config.r_T
        self.r_F = config.r_F
        self.r_M = config.r_M_init
        self.pop = pop# config.Pop_arrival
        
        self.T_Pref_i = config.T_Pref_max
        self.F_Req_i = 0
        self.T_Req_i = 0
        self.fisher=False 
        self.tr99 = self.T_Pref_i* config.T_Req_pP* self.pop * config.H_equ
        self.f99 = (1-self.T_Pref_i)* config.F_Req_pP* self.pop * config.H_equ

        self.A_F_i = np.array([]).astype(int)
        self.F_PI_i = np.array([])

        self.tree_fill = 1
        self.farming_fill = 1

        self.h_i=1.0
        self.H_i = 1.0

        self.alpha_W = config.alpha_W#0.3
        eta_i = (self.T_Pref_i * config.alpha_T + (1-self.T_Pref_i)*config.alpha_F)/(config.alpha_T+config.alpha_F)
        self.alpha_T_prime = config.alpha_T * self.T_Pref_i/eta_i#0.2
        self.alpha_D = config.alpha_D#0.3
        self.alpha_F_prime = config.alpha_F * (1-self.T_Pref_i)/eta_i# 0.2
        self.alpha_G = config.alpha_G

        self.penalty = 0
        return 
    
    def calc_T_Req(self):
        self.T_Req_i = np.ceil((config.T_Req_pP*self.pop * self.T_Pref_i)).astype(int)         #
        self.tr99 = self.T_Pref_i* config.T_Req_pP* self.pop * config.H_equ
        return 
    def calc_F_Req(self):
        if self.fisher:
            self.F_Req_i=0
        else:
            self.F_Req_i = (self.pop * config.F_Req_pP *  (1-self.T_Pref_i))
        self.f99 = (1-self.T_Pref_i)* config.F_Req_pP* self.pop * config.H_equ

    #def calc_new_tree_pref_linear(self):
    #    self.T_Pref_i = np.clip( config.EI.tree_density[self.mv_inds].sum()/config.EI.carrying_cap[self.mv_inds].sum(), config.T_Pref_min, config.T_Pref_max)
    #    return 

    def calc_new_tree_pref(self):
        x= config.EI.T_c[ config.EI.C_T_c[self.c_i]].sum()/config.EI.carrying_cap[ config.EI.C_T_c[self.c_i]].sum()
        self.T_Pref_i = config.f_T_Pref(x)
        if self.fisher:
            self.T_Pref_i = max(self.T_Pref_i, config.T_Pref_fisher_min)
        eta_i = (self.T_Pref_i * config.alpha_T + (1-self.T_Pref_i)*config.alpha_F)/(config.alpha_T+config.alpha_F)
        self.alpha_T_prime = config.alpha_T * self.T_Pref_i/eta_i#0.2
        self.alpha_F_prime = config.alpha_F * (1-self.T_Pref_i)/eta_i# 0.2
        return 


    # Helper Function
    #def euclid_distance_1vec1ag(self, m,ag):
    #    return np.linalg.norm(m-np.array([ag.x, ag.y]))


    def split_household(self, t):
        # IF Population TOO LARGE; SPlit AGENT

        # 2* config.children pop = 24 is the minimum people to split!! Otherwise I could perhaps have a split in which too few people remain to make sens
        if self.pop > np.random.gamma(config.Split_k,config.Split_theta) + 2*config.SplitPop:
            child = copy(self)
            child.index = config.index_count
            config.index_count+=1
            config.EI.agNr_c[child.c_i]+=1
            child.pop = int(config.SplitPop)
            child.fisher=False
            child.calc_new_tree_pref()
            child.calc_F_Req()
            child.calc_T_Req()
            child.A_F_i=np.array([]).astype(int) # Note, if child doesn't have any garden, we also do not clear space when moving
            child.F_PI_i= np.array([])
            move(child, t)

            # Adjust parent agent and reduce its agriculture.
            self.pop -= int(config.SplitPop)
            #self.calc_new_tree_pref() # will do in update.py
            #self.calc_F_Req()
            #self.calc_T_Req()
            config.agents.append(child)
        return 

    def remove_unncessary_agric(self):
        reduceAgric = (np.sum(self.F_PI_i) - self.F_Req_i) #self.AgriNeed*(self.pop-config.init_pop)/self.pop)
        for site in  self.A_F_i[np.argsort(self.F_PI_i)]:
            if reduceAgric>0:
                reduceAgric -= config.EI.F_pi_c[site]
                if reduceAgric>=0:
                    config.EI.A_F_c[site]-=1
                    self.F_PI_i = np.delete(self.F_PI_i, np.where(self.A_F_i==site)[0][0]) # only delete first occurence
                    self.A_F_i = np.delete(self.A_F_i, np.where(self.A_F_i==site)[0][0]) # only delete first occurence
                else:
                    break
            else:
                break
        return 

    def remove_agent(self):
        # CHECK IF AGENT IS STILL LARGE ENOUGH OR DIES
        #config.deaths+= self.pop
        config.EI.agNr_c -=1
        config.EI.pop_c[self.c_i] -= self.pop
        if self.fisher: config.FisherAgents-=1
        for i in self.A_F_i:
            config.EI.A_F_c[i]-=1
        self.A_F_i=[]
        self.F_PI_i=[]
        config.agents.remove(self)
        print("Agent ",self.index," died.")
        
        # REfugess
        potential_hhs = [ag for ag in config.agents if np.linalg.norm(np.array([ag.x-self.x, ag.y-self.y]))<self.r_M]
        if len(potential_hhs)>0:
            acceptors = np.random.choice(potential_hhs, size=self.pop, replace =True) 
            for a in acceptors:
                a.pop +=1
                config.EI.pop_c[a.c_i]+=1
        else:
            config.excess_deaths += self.pop
        return False # i.e. not survived
        
    def population_change(self, H,t):
        g = config.g_of_H(H)
        rands = np.random.random(self.pop)
        if g>=1:
            popchange = np.sum(rands<(g-1))
            config.excess_births +=abs(popchange)
        elif g<1:
            popchange = -np.sum(rands < (1-g))
            config.excess_deaths+= abs(popchange)
        self.pop += popchange
        config.EI.pop_c[self.c_i]+=popchange

        self.split_household(t)
        if self.pop < config.pop_agent_min:
            survived = self.remove_agent()
        else:
            survived = True
            self.remove_unncessary_agric()

        return survived


def init_agents():#N_agents, init_option, agent_type, map_type):
    config.agents=[]
    distances = scipy.spatial.distance.cdist(
        np.array([config.EI.vec_c[config.EI.AnakenaBeach_ind]]), 
        config.EI.vec_c)[0]
    possible_triangles = np.where(distances < config.r_M_arrival)[0]

    for _ in range(config.N_agents_arrival):
      
        x,y = config.EI.vec_c[config.EI.AnakenaBeach_ind]

        ag = agent(x, y, int(config.Pop_arrival/config.N_agents_arrival))
        config.EI.agNr_c[config.EI.AnakenaBeach_ind]+=1
        config.EI.pop_c[config.EI.AnakenaBeach_ind]+=ag.pop
        ag.c_i= config.EI.AnakenaBeach_ind

        #possible_triangles = np.arange(config.EI.N_els)
        move(ag, config.t_arrival, FirstSettlementAnakenaTriangles=[True, possible_triangles])
        #ag.triangle_ind = ind
        #ag.triangle_midpoint = m_meter
        #config.EI.agentOccupancy[ind]+=1
        #config.EI.populationOccupancy[ind] += ag.pop
        ag.calc_new_tree_pref()
        ag.calc_T_Req()
        ag.calc_F_Req()
        config.agents.append(ag)  # add agent to list
    return 

