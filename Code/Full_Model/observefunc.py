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
font = {'font.size':10}
plt.rcParams.update(font)
import matplotlib.colors as colors

from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys, os

def observe(t):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1,1,1,fc='aquamarine')

    fig,ax = observe_density(t,ax = ax,fig = fig, save=False)
    fig,ax = config.EI.plot_agricultureSites(ax=ax, fig=fig, save=False, CrossesOrFacecolor="Crosses")

    ax.set_title("Time "+str(t))
    fig.tight_layout()
    plt.savefig(config.folder+"map_time"+str(t)+".png")
    plt.close()


def observe_density(t, fig=None, ax=None, save=True, hist=False, folder=config.folder, specific_ag_to_follow=None):# agents, EI, t, agent_type, folder):


    if ax==None:
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)        

    fig, ax = config.EI.plot_TreeMap_and_hist(ax=ax, fig=fig, hist=hist,folder=folder)
    #ax.tripcolor(config.EI.points_EI_km[:,0], config.EI.corners['upper_left'][1] - config.EI.points_EI_km[:,1], 
    #    config.EI.EI_triangles, facecolors=np.array([1 for _ in range(config.EI.N_els)]), vmin = 0, vmax = 1, cmap="Blues", alpha=1)
    #
    #plot = ax.tripcolor(config.EI.points_EI_km[:,0], config.EI.corners['upper_left'][1] - config.EI.points_EI_km[:,1], 
    #  config.EI.EI_triangles, facecolors=config.EI.tree_density, vmin = 0, vmax = np.max(config.EI.carrying_cap), cmap="Greens", alpha=1)
    ##cbaxes = fig.add_axes([0.9, 0.0, 0.03, 1.0]) 
    #fig.colorbar(plot)#, cax = cbaxes)  
    ax = plot_agents_on_top(ax, t,ncdf=False, specific_ag_to_follow=specific_ag_to_follow)

    if save:
        ax.set_title("Time "+str(t))
        fig.tight_layout()
        plt.savefig(config.folder+"map_time"+str(t)+".png")
        plt.close()
    return fig,ax



def follow_ag0(ag):
    ''' plot circle around ag0 with resource and move search radius'''
    move_circle = plt.Circle([ag.x, config.EI.corners['upper_left'][1]-ag.y], 
        radius = ag.moving_radius, clip_on=True, color='black', alpha=0.2)
    tree_circle = plt.Circle([ag.x, config.EI.corners['upper_left'][1]-ag.y], 
        radius = ag.tree_search_radius, clip_on=True, color='black', alpha=0.35)
    agric_circle = plt.Circle([ag.x, config.EI.corners['upper_left'][1]-ag.y], 
        radius = ag.agriculture_radius, clip_on=True, color='black', alpha=0.5)
    return move_circle, tree_circle, agric_circle



# def plot_dynamics(N_time_steps, ncdf_info=None):#folder, stats, N_time_steps):
#     if not ncdf_info==None: # then i am reading from the .ncdf stats file
#         nr_agents, nr_pop, nr_deaths, nr_trees =  ncdf_info['stats'] 
#         #agent_type = ncdf_info['agent_type']
#         folder = ncdf_info['folder']
#     else:
#         nr_agents, nr_pop, nr_deaths, nr_trees = [config.nr_agents, 
#                 config.nr_pop, config.nr_deaths, config.nr_trees]
#         folder=config.folder
#     # PLOTTING
#     rcParam = {'font.size':20}
#     plt.rcParams.update(rcParam)
#     fig = plt.figure(figsize=(16,9))
#     ax_agent = fig.add_subplot(111)
#     ax_tree = ax_agent.twinx()
#     ax_agent.plot(nr_agents[:], '-',lw=3, color='blue')
#     ax_agent.plot(nr_deaths[:], '-', lw=3, color='black')
#     ax_tree.plot(nr_trees[:], '--', lw=3, color='green')
    
#     ax_pop = ax_agent.twinx()   
#     ax_pop.plot(nr_pop[:], '-',lw=3, color='cyan')
#     ax_pop.set_ylabel("Nr of Pop")
#     ax_pop.tick_params(axis="y", colors='cyan')
#     ax_pop.yaxis.label.set_color("cyan")
    
#     ax_tree.set_ylabel("Nr of trees")
#     ax_agent.set_ylabel(r"Nr of agents  and  deaths (black)")
#     ax_agent.tick_params(axis="y", colors='blue')
#     ax_agent.yaxis.label.set_color("blue")
#     ax_tree.tick_params(axis="y", colors='green')
#     ax_tree.yaxis.label.set_color("green")
#     ax_agent.set_xlabel("Time")
#     ax_agent.set_ylim(0,)
#     ax_tree.set_ylim(0,)
#     plt.savefig(folder+"run_counts.png")
#     plt.close()

#     if ncdf_info==None:
#         import imageio
#         images = []
#         filenames=[folder+"map_time"+str(t)+".png" for t in range(N_time_steps)]
#         for filename in filenames:
#             images.append(imageio.imread(filename))
#         imageio.mimsave(folder+'movie.gif', images, duration=0.4)



def plot_movingProb(t,ag, newpos):
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(3,3,1,fc='aquamarine')
    rcParams={'font.size':10}
    plt.rcParams.update(rcParams)

    fig, ax = observe_density(t, fig=fig, ax=ax, save=False, hist=False, folder="", specific_ag_to_follow=ag)
    fig,ax = config.EI.plot_agricultureSites(ax=ax, fig=fig, save=False, CrossesOrFacecolor="Crosses")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    ax.set_title("Agent "+str(ag.index))
    move_circle, tree_circle, agric_circle = follow_ag0(ag)
    ax.add_artist(move_circle)
    ax.add_artist(tree_circle)
    ax.add_artist(agric_circle)

    ax.set_xlim(max(ag.x-ag.moving_radius*1.1, 0), min(ag.x+ag.moving_radius*1.1, config.EI.corners["upper_right"][0]))
    ax.set_ylim(max(config.EI.corners['upper_left'][1]-ag.y-ag.moving_radius*1.1,0), min(config.EI.corners['upper_left'][1]-ag.y+ag.moving_radius*1.1, config.EI.corners["upper_right"][1]))
       
    ax.set_aspect(1)
    ax.plot(newpos[0], config.EI.corners['upper_left'][1] - newpos[1], "o", markersize = 8,color='magenta')
    #divider = make_axes_locatable(plt.gca())
    #cax = divider.append_axes("right", "5%", pad="3%")
    #plt.colorbar(plot, cax =cax) 



    #ax_text = fig.add_subplot(3,3,9, fc="white")#, xticklabels=[], yticklabels=[], xticks=[], yticks=[], fc="white")
    #s= "Weights"+"\n"+r"$\alpha_w=$"+str(ag.alpha_w)+"\n"+r"$\alpha_t=$"+str(ag.alpha_t)+"\n"+r"$\alpha_a=$"+str(ag.alpha_a)+"\n"+r"$\alpha_p=$"+str(ag.alpha_p)+"\n"+r"$\alpha_m=$"+str(ag.alpha_m)
    #ax_text.text(0.4,0.5, s, horizontalalignment='center', verticalalignment='center', fontsize=10)
    #ax_text.set_xticklabels([])
    #ax_text.set_yticklabels([])

    penalties = [config.AG0_total_penalties, config.AG0_MovingProb, config.AG0_water_penalties, config.AG0_tree_penalty, config.AG0_agriculture_penalty, config.AG0_pop_density_penalty, config.AG0_map_penalty, config.AG0_nosettlement_zones]
    penalties_key = ["total_penalties", "MovingProb","water_penalties", "tree_penalty", "agriculture_penalty", "pop_density_penalty", "Map_Penalty","Allowed_settlements"]
    for k, penalty,key in zip([2,3,4,5,6,7,8,9], penalties, penalties_key):
        ax2  = fig.add_subplot(3,3,k, fc="aquamarine")
        #ax2.tripcolor(config.EI.points_EI_km[:,0], config.EI.corners['upper_left'][1] - config.EI.points_EI_km[:,1], 
        #        config.EI.EI_triangles, facecolors=np.array([1 for _ in range(config.EI.N_els)]), vmin = 0, vmax = 1, cmap="binary", alpha=1)

        ax2.set_xlim(max(ag.x-ag.moving_radius*1.1, 0), min(ag.x+ag.moving_radius*1.1, config.EI.corners["upper_right"][0]))
        ax2.set_ylim(max(0,config.EI.corners['upper_left'][1]-ag.y-ag.moving_radius*1.1), min(config.EI.corners['upper_left'][1]-ag.y+ag.moving_radius*1.1, config.EI.corners["upper_right"][1]))
        ax2.set_aspect(1)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])

        #for p in range(config.EI.N_els):
        #    if p in config.AG0_mv_inds:
        #        prob[p] = penalty[np.where(config.AG0_mv_inds==p)[0]]
        if k==9:
            #cmap = colors.ListedColormap(['black', 'white','red','green', 'yellow'])
            #boundaries = [-0.1, 0.5, 1.5, 2.5, 3.5, 5.5,6.5,7.5]
            #norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
            # 0 none
            # 1 Y:S, N:T,A
            # 2 Y:A, N:S,T
            # 3 Y:S,A N:T
            # 4  ....
            #liste = {'#00000000':0,'#000000ff':1,'#0000ff00':2,'#00ff0000':3,'#ff000000':4,'#ffffffff':5}#,'#00ff00ff', '#00ffffff', '#ffffffff', '#00000000']
            #labels = ["Not allowed", "Not (slope cond)", "Not (pop cond)", "Not (tree cond)", "No (agric cond)", "Allowed"]
            #cmap = mpl.colors.ListedColormap(liste.keys())
            #boundaries = np.arange(-0.5, 5.5, step=1)
            #norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)#

            colors = np.array([0 for i in range(config.EI.N_els)])
            #colors[config.AG0_mv_inds] = [liste[mpl.colors.to_hex(penalty[:,i], keep_alpha=True)] for i in range(penalty.shape[1])]
            colors[config.AG0_mv_inds] = [1 if np.sum(penalty[:,i])==4 else 0 for i in range(penalty.shape[1])]
            plot  = ax2.tripcolor(config.EI.points_EI_km[:,0], config.EI.corners['upper_left'][1] - config.EI.points_EI_km[:,1], 
                config.EI.EI_triangles, facecolors=colors, vmin = 0, cmap='binary_r',alpha=1)
        else:
            prob = np.zeros([config.EI.N_els])
            prob[config.AG0_mv_inds] = penalty

            if k==3:
                cmap="hot_r"
                vmax=np.max(prob).clip(min=0.01)
            elif k ==2:
                cmap="Reds"
                vmax=np.max(prob).clip(min=0.01)
            else:
                cmap="Reds"
                vmax=1
            plot  = ax2.tripcolor(config.EI.points_EI_km[:,0], config.EI.corners['upper_left'][1] - config.EI.points_EI_km[:,1], 
                config.EI.EI_triangles, facecolors=prob,vmin=0,vmax=vmax, cmap=cmap, alpha=1)
    

        if not all(config.AG0_mv_inds==ag.mv_inds):
            print("Error in Plot Penalties")
        ax2.plot(newpos[0], config.EI.corners['upper_left'][1]-newpos[1], "o", markersize = 8,color='magenta', fillstyle="none")

    
        
        ax2.set_title(key)
        #print(np.min(prob),np.max(prob))
        move_circle, tree_circle, agric_cirle = follow_ag0(ag)
        ax2.add_artist(move_circle)
        ax2.add_artist(tree_circle)
        ax2.add_artist(agric_cirle)
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        plt.colorbar(plot, cax =cax) 

    fig.tight_layout()
    #fig.colorbar(p)
    plt.savefig(config.folder+"Penalties_AG"+str(ag.index)+"_t="+str(t)+".png", bbox_inches='tight')











######################################################################
###########        PLOT FUNCTIONS LATER                 ##############
######################################################################
def plot_agents_on_top(ax, t, ncdf=False, data=None, specific_ag_to_follow=None, color = 'purple'):
    if ncdf==False:
        agent_points_np = np.array([[ag.x, ag.y, ag.pop] for ag in config.agents])# subtracting ag.y from the upper left corner gives a turned picture (note matrix [0][0] is upper left)
        agent_size = agent_points_np[:,2]
    else:
        agent_points_np = data.PosAgents.sel(time=t).to_series().dropna()
        agent_size = data.SizeAgents.set(time=t).to_series().dropna()
    
    color = [((1-ag.treePref), 0, 1,1) for ag in config.agents]

    if len(config.agents)>0:
        ax.scatter(agent_points_np[:,0], config.EI.corners['upper_left'][1] - agent_points_np[:,1], s=agent_size, color=color)#'purple')
        if specific_ag_to_follow==None:
            specific_ag_to_follow = config.agents[0]
        move_circle, tree_circle, agric_cirle = follow_ag0(specific_ag_to_follow)
        ax.add_artist(move_circle)
        ax.add_artist(tree_circle)
        ax.add_artist(agric_cirle)
    return ax