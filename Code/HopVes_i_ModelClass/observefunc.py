#!/usr/bin/env python

# PETEr STEIGLECHNER, 9.3.2020, 
# 
''' This File contains the 
 - observe function where treees are agents (for HopVes_i Base and _HHet)
 - observe_density function where trees are just a number associated with a triangle on Easter Island (HopVes_i_spatExt, +_hh_ext)
 - follow_ag0
 '''

import config

import matplotlib as mpl 
import numpy as np
import matplotlib.pyplot as plt
font = {'font.size':20}
plt.rcParams.update(font)

import sys, os

def observe(t):
    if config.map_type=="EI":
        observe_density(t)
    elif config.map_type=="unitsquare":
        observe_classical(t)
    return 

#  Observe  = Plot the Map
def observe_classical(t):  #agents, trees, t, agent_type, folder):
    ''' Plot a map of histogram of tree agents and the agent's location at time t, 
        agents_type="individual" or "household"
    '''
    tree_points = []
    for tree in config.trees:
        tree_points.append([tree.x, tree.y])
    tree_points_np = np.array(tree_points)
    agent_points = []
    for ag in config.agents:
        if config.agent_type=="household":
            agent_points.append([ag.x, ag.y, 2*ag.pop])
        elif config.agent_type=="individual":
            agent_points.append([ag.x, ag.y, 20])
        else:
            print("ERROR: Choose right agent_type")
    agent_points_np = np.array(agent_points)
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)

    if len(config.trees)>0:
        bins=20
        plt.hist2d(tree_points_np[:,0], 
                tree_points_np[:,1], 
                bins=bins, 
                cmap='Greens', 
                vmin=0, 
                vmax = config.N_trees/(bins**2),#config.nr_trees[0]/(bins**2),
                range = [[0,1],[0,1]]
                )
    if len(config.agents)>0:
        ax.scatter(agent_points_np[:,0], agent_points_np[:,1], s=agent_points_np[:,2], color='blue')
    ax.set_title("Time "+str(t))
    if len(config.trees)>0: plt.colorbar()
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    fig.tight_layout()
    plt.savefig(config.folder+"map_time"+str(t)+".png")
    plt.close()
    return 



def observe_density(t):# agents, EI, t, agent_type, folder):
    agent_points = []
    for ag in config.agents:
        # subtracting ag.y from the upper left corner gives a turned picture (note matrix [0][0] is upper left)
        if config.agent_type=="household":
            agent_points.append([ag.x, config.EI.corners['upper_left'][1]-ag.y, 2*ag.pop])
        elif config.agent_type=="individual":
            agent_points.append([ag.x, config.EI.corners['upper_left'][1]-ag.y, 20])
        else:
            config.wrongAgentType()
    agent_points_np = np.array(agent_points)

    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111)
    ax.tripcolor(config.EI.points_EI_m[:,0], config.EI.corners['upper_left'][1] - config.EI.points_EI_m[:,1], 
        config.EI.EI_triangles, facecolors=np.array([1 for _ in range(config.EI.N_els)]), vmin = 0, vmax = 1, cmap="Blues", alpha=1)
    
    plot = ax.tripcolor(config.EI.points_EI_m[:,0], config.EI.corners['upper_left'][1] - config.EI.points_EI_m[:,1], 
      config.EI.EI_triangles, facecolors=config.EI.tree_density, vmin = 0, vmax = np.max(config.EI.carrying_cap), cmap="Greens", alpha=0.8)
    #cbaxes = fig.add_axes([0.9, 0.0, 0.03, 1.0]) 
    fig.colorbar(plot)#, cax = cbaxes)  
    if len(config.agents)>0:
        ax.scatter(agent_points_np[:,0], agent_points_np[:,1], s=agent_points_np[:,2], color='blue')
        move_circle, res_circle = follow_ag0(config.agents[0])
        ax.add_artist(move_circle)
        ax.add_artist(res_circle)
    ax.set_title("Time "+str(t))
    fig.tight_layout()
    plt.savefig(config.folder+"map_time"+str(t)+".png")
    plt.close()
    return 

def follow_ag0(ag):
    ''' plot circle around ag0 with resource and move search radius'''
    move_circle = plt.Circle([ag.x, config.EI.corners['upper_left'][1]-ag.y], 
        radius = ag.moving_radius, clip_on=True, color='black', alpha=0.2)
    res_circle = plt.Circle([ag.x, config.EI.corners['upper_left'][1]-ag.y], 
        radius = ag.resource_search_radius, clip_on=True, color='black', alpha=0.4)
    return move_circle, res_circle



def plot_dynamics(N_time_steps, ncdf_info=None):#folder, stats, N_time_steps):
    if not ncdf_info==None: # then i am reading from the .ncdf stats file
        nr_agents, nr_pop, nr_deaths, nr_trees =  ncdf_info['stats'] 
        agent_type = ncdf_info['agent_type']
        folder = ncdf_info['folder']
    else:
        nr_agents, nr_pop, nr_deaths, nr_trees = [config.nr_agents, 
                config.nr_pop, config.nr_deaths, config.nr_trees]
        agent_type=config.agent_type 
        folder=config.folder
    # PLOTTING
    rcParam = {'font.size':20}
    plt.rcParams.update(rcParam)
    fig = plt.figure(figsize=(16,9))
    ax_agent = fig.add_subplot(111)
    ax_tree = ax_agent.twinx()
    ax_agent.plot(nr_agents[:], '-',lw=3, color='blue')
    ax_agent.plot(nr_deaths[:], '-', lw=3, color='black')
    ax_tree.plot(nr_trees[:], '--', lw=3, color='green')
    
    if agent_type=="household":
        ax_pop = ax_agent.twinx()   
        ax_pop.plot(nr_pop[:], '-',lw=3, color='cyan')
        ax_pop.set_ylabel("Nr of Pop")
        ax_pop.tick_params(axis="y", colors='cyan')
        ax_pop.yaxis.label.set_color("cyan")
    
    ax_tree.set_ylabel("Nr of trees")
    ax_agent.set_ylabel(r"Nr of agents  and  deaths (black)")
    ax_agent.tick_params(axis="y", colors='blue')
    ax_agent.yaxis.label.set_color("blue")
    ax_tree.tick_params(axis="y", colors='green')
    ax_tree.yaxis.label.set_color("green")
    ax_agent.set_xlabel("Time")
    ax_agent.set_ylim(0,)
    ax_tree.set_ylim(0,)
    plt.savefig(folder+"run_counts.png")


    if ncdf_info==None:
        import imageio
        images = []
        filenames=[folder+"map_time"+str(t)+".png" for t in range(N_time_steps)]
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave(folder+'movie.gif', images, duration=0.4)



