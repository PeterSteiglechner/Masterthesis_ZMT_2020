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
from mpl_toolkits.axes_grid1.colorbar import colorbar
from matplotlib.colors import LinearSegmentedColormap

import sys, os

def observe(t, fig=None, ax=None, specific_ag_to_follow=None, save=True):
    if fig==None:
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(1,1,1,fc='aquamarine')

    #fig,ax = observe_density(t,ax = ax,fig = fig, save=False)
    #fig,ax = config.EI.plot_agricultureSites(ax=ax, fig=fig, save=False, CrossesOrFacecolor="Crosses")

    fig, ax, divider, _ =   plot_TreeMap_only(fig, ax, t, data=None, ncdf=False, save = save)
    ax, divider, _  = plot_agricultureSites_onTop(ax, divider, t, data=None, ncdf=False, save =save)
    if len(config.agents)>0:
        ax = plot_agents_on_top(ax, t, ncdf=False, data=None, specific_ag_to_follow=specific_ag_to_follow, color = 'purple')

    ax.set_title("Time "+str(t))

    if save==True:
        #fig.tight_layout()
        plt.savefig(config.folder+"map_time"+str(t)+".png")
        plt.close()
    return fig, ax


def follow_ag0(x,y,mr, tr, ar):
    ''' plot circle around ag0 with resource and move search radius'''
    move_circle = plt.Circle([x, config.EI.corners['upper_left'][1]-y], 
        radius = mr, clip_on=True, color='black', alpha=0.2)
    tree_circle = plt.Circle([x, config.EI.corners['upper_left'][1]-y], 
        radius = tr, clip_on=True, color='black', alpha=0.35)
    agric_circle = plt.Circle([x, config.EI.corners['upper_left'][1]-y], 
        radius = ar, clip_on=True, color='black', alpha=0.5)
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




def plot_movingProb(ToSavePenalties):
    [AG_total_penalties,
                AG_water_penalties,
                AG_tree_penalty,
                AG_pop_density_penalty,
                AG_agriculture_penalty,
                AG_map_penalty,
                AG_nosettlement_zones,
                AG_MovingProb,
                AG_Pos,
                AG_NewPos,
                AG_InitialTriangle,
                _,#AG_NewPosTriangle,
                AG_MovingRad,
                AG_TPref,
                AG_Pop,
                _,#AG_H
                t,
                AG_index] = [ToSavePenalties[key] for key in ToSavePenalties.keys()]
    plt.rcParams.update({"font.size":15})
    fig = plt.figure(figsize=(10*3,6*3))
    ax = fig.add_subplot(3,3,1,fc='aquamarine')
    #rcParams={'font.size':10}
    #plt.rcParams.update(rcParams)

    fig, ax = observe(t, fig=fig, ax=ax, specific_ag_to_follow=None, save=False) #(t, fig=fig, ax=ax, save=False, hist=False, folder="", specific_ag_to_follow=ag)
    #fig,ax = config.EI.plot_agricultureSites(ax=ax, fig=fig, save=False, CrossesOrFacecolor="Crosses")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    ax.set_title("Agent "+str(AG_index))


    move_circle, tree_circle, agric_circle = follow_ag0(AG_Pos[0], AG_Pos[1], AG_MovingRad, config.EI.tree_search_radius, config.EI.agriculture_radius)
    if AG_MovingRad <100: ax.add_artist(move_circle)
    ax.add_artist(tree_circle)
    ax.add_artist(agric_circle)

    ax.set_xlim(max(AG_Pos[0]-AG_MovingRad*1.1, 0), min(AG_Pos[0]+AG_MovingRad*1.1, config.EI.corners["upper_right"][0]))
    ax.set_ylim(max(config.EI.corners['upper_left'][1]-AG_Pos[1]-AG_MovingRad*1.1,0), min(config.EI.corners['upper_left'][1]-AG_Pos[1]+AG_MovingRad*1.1, config.EI.corners["upper_right"][1]))
       
    ax.set_aspect(1)
    #ax.plot(newpos[0], config.EI.corners['upper_left'][1] - newpos[1], "o", markersize = 8,color='magenta')
    #divider = make_axes_locatable(plt.gca())
    #cax = divider.append_axes("right", "5%", pad="3%")
    #plt.colorbar(plot, cax =cax) 



    #ax_text = fig.add_subplot(3,3,9, fc="white")#, xticklabels=[], yticklabels=[], xticks=[], yticks=[], fc="white")
    #s= "Weights"+"\n"+r"$\alpha_w=$"+str(ag.alpha_w)+"\n"+r"$\alpha_t=$"+str(ag.alpha_t)+"\n"+r"$\alpha_a=$"+str(ag.alpha_a)+"\n"+r"$\alpha_p=$"+str(ag.alpha_p)+"\n"+r"$\alpha_m=$"+str(ag.alpha_m)
    #ax_text.text(0.4,0.5, s, horizontalalignment='center', verticalalignment='center', fontsize=10)
    #ax_text.set_xticklabels([])
    #ax_text.set_yticklabels([])
    if AG_MovingRad<100:
        mv_inds = config.EI.LateMovingNeighbours_of_triangles[AG_InitialTriangle]
    else:
        mv_inds = np.arange(config.EI.N_els)
    #penalties = [config.AG0_total_penalties, config.AG0_MovingProb, config.AG0_water_penalties, config.AG0_tree_penalty, config.AG0_agriculture_penalty, config.AG0_pop_density_penalty, config.AG0_map_penalty, config.AG0_nosettlement_zones]
    penalties = [AG_total_penalties, AG_MovingProb, AG_water_penalties, AG_tree_penalty, AG_agriculture_penalty, AG_pop_density_penalty, AG_map_penalty, AG_nosettlement_zones]
    penalties_key = ["total_penalties", "MovingProb","water_penalties", "tree_penalty", "agriculture_penalty", "pop_density_penalty", "Map_Penalty","Allowed_settlements"]
    for k, penalty,key in zip([2,3,4,5,6,7,8,9], penalties, penalties_key):
        ax2  = fig.add_subplot(3,3,k, fc="aquamarine")
        #ax2.tripcolor(config.EI.points_EI_km[:,0], config.EI.corners['upper_left'][1] - config.EI.points_EI_km[:,1], 
        #        config.EI.EI_triangles, facecolors=np.array([1 for _ in range(config.EI.N_els)]), vmin = 0, vmax = 1, cmap="binary", alpha=1)

        ax2.set_xlim(max(AG_Pos[0]-AG_MovingRad*1.1, 0), min(AG_Pos[0]+AG_MovingRad*1.1, config.EI.corners["upper_right"][0]))
        ax2.set_ylim(max(0,config.EI.corners['upper_left'][1]-AG_Pos[1]-AG_MovingRad*1.1), min(config.EI.corners['upper_left'][1]-AG_Pos[1]+AG_MovingRad*1.1, config.EI.corners["upper_right"][1]))
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
            colors[mv_inds] = [1 if np.sum(penalty[:,i])==4 else 0 for i in range(penalty.shape[1])]
            plot  = ax2.tripcolor(config.EI.points_EI_km[:,0], config.EI.corners['upper_left'][1] - config.EI.points_EI_km[:,1], 
                config.EI.EI_triangles, facecolors=colors, vmin = 0, cmap='binary_r',alpha=1)
        else:
            prob = np.zeros([config.EI.N_els])-np.spacing(0.0)
            prob[mv_inds] = penalty

            if k==3:
                #cmap="hot_r"
                cmap = plt.get_cmap('magma_r')
                cmap.set_under('gray') 
                vmin = 0.0
                vmax=np.max(prob).clip(max=0.01)
            elif k ==2:
                cmap = plt.get_cmap("Reds")
                cmap.set_under('gray') 
                vmax=np.max(prob).clip(min=0.01)
                vmin= 0.0                
            else:
                cmap = plt.get_cmap("Reds")
                cmap.set_under('gray') 
                vmin = 0.0
                vmax=1
                if k==6:
                    vmin = -1.0# np.min(penalty)
                    prob = np.zeros([config.EI.N_els])-2.0-np.spacing(0.0)
                    prob[mv_inds] = penalty
                    vmax = 1
                    colorsneg = plt.get_cmap("Blues_r")(np.linspace( 0.0,1, 128))
                    colorspos = plt.get_cmap("Reds")(np.linspace(0, 1.0, 128))
                    colorsstack = np.vstack((colorsneg, colorspos))
                    cmap = LinearSegmentedColormap.from_list('my_colormap', colorsstack)
                    cmap.set_under('gray') 

            plot  = ax2.tripcolor(config.EI.points_EI_km[:,0], config.EI.corners['upper_left'][1] - config.EI.points_EI_km[:,1], 
                config.EI.EI_triangles, facecolors=prob,vmin=vmin,vmax=vmax, cmap=cmap, alpha=1)


                

        if not all(config.AG0_mv_inds==AG_MovingRad):
            print("Error in Plot Penalties")
        ax2.plot(AG_NewPos[0], config.EI.corners['upper_left'][1]-AG_NewPos[1], "o", markersize = int(AG_Pop/3),color=((1-AG_TPref[0]), 0, 1,1), fillstyle="none")
        ax2.plot(AG_Pos[0], config.EI.corners['upper_left'][1]-AG_Pos[1], "o", markersize = int(AG_Pop/3) ,color=((1-AG_TPref[0]), 0, 1,1))
        
    
        
        ax2.set_title(key)
        #print(np.min(prob),np.max(prob))
        #move_circle, tree_circle, agric_cirle = follow_ag0(ag)
        #ax2.add_artist(move_circle)
        #ax2.add_artist(tree_circle)
        #ax2.add_artist(agric_cirle)
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        plt.colorbar(plot, cax =cax) 

    #fig.tight_layout()
    #fig.colorbar(p)
    plt.savefig(config.folder+"Penalties_AG"+str(AG_index)+"_t="+str(t)+".png", bbox_inches='tight')











######################################################################
###########        PLOT FUNCTIONS LATER                 ##############
######################################################################
def plot_agents_on_top(ax, t, ncdf=False, data=None, specific_ag_to_follow=None, color = 'purple'):
    if ncdf==False:
        agent_points_np = np.array([[ag.x, ag.y, ag.pop] for ag in config.agents])# subtracting ag.y from the upper left corner gives a turned picture (note matrix [0][0] is upper left)
        agent_size = agent_points_np[:,2]
    else:
        agent_points_np = data.PosAgents.sel(time=t).to_series().dropna()
        agent_size = data.SizeAgents.sel(time=t).to_series().dropna()
        agent_points_np = np.array([agent_points_np[::2], agent_points_np[1::2]]).T
    
    if ncdf == False:
        color = [((1-ag.treePref), 0, 1,1) for ag in config.agents]
    else:
        color = color

    if len(agent_points_np)>0:
        ax.scatter(agent_points_np[:,0], config.EI.corners['upper_left'][1] - agent_points_np[:,1], s=agent_size, color=color)#'purple')

        if ncdf==False:
            if not specific_ag_to_follow==None:
                #specific_ag_to_follow = config.agents[0]
                a = specific_ag_to_follow
                move_circle, tree_circle, agric_cirle = follow_ag0(a.x, a.y, a.moving_radius, a.tree_search_radius, a.agriculture_radius)
                if not a.moving_radius==100: ax.add_artist(move_circle)
                ax.add_artist(tree_circle)
                ax.add_artist(agric_cirle)
    return ax





def plot_agricultureSites_onTop(ax, divider, t, data=None, ncdf=False, save=True):
    if ncdf:
        face_colors = data.agriculture.sel(time=t)* 1/(config.EI.EI_triangles_areas * config.km2_to_acre)  #(config.EI.nr_highqualitysites+config.EI.nr_lowqualitysites).clip(min=0.1)
    else:
        face_colors = config.EI.agriculture* 1/(config.EI.EI_triangles_areas * config.km2_to_acre)  #(config.EI.nr_highqualitysites+config.EI.nr_lowqualitysites).clip(min=0.1)
    if any(face_colors>1):
        print("Error in plot agricultureSites")
    colors = [(1,0,0,0), (1,0,0,1)]
    #n_bin = 1# np.max(config.EI.agriculture)
    customCmap = LinearSegmentedColormap.from_list("agriccmap",colors,N=100)

    #plot_agricultureSites_onTop(ax=ax, fig=fig)    
    AgricPlot = ax.tripcolor(config.EI.points_EI_km[:,0], config.EI.corners['upper_left'][1] - config.EI.points_EI_km[:,1], config.EI.EI_triangles, facecolors=face_colors, cmap=customCmap, alpha=None, vmax=1, vmin=0)

    if save:
        cax2 = divider.append_axes("right", "5%", pad="14%")
        cb2 = colorbar(AgricPlot, cax =cax2)     
        cb2.set_label_text("Agriculture Fraction of Triangle")
    return ax, divider, AgricPlot


def plot_TreeMap_only(fig, ax, t, data=None, ncdf=False, save=True):
    if ncdf:
        face_colors =data.tree_density.sel(time=t)
    else:
        face_colors = config.EI.tree_density
    treePlot = ax.tripcolor(config.EI.points_EI_km[:,0], config.EI.corners['upper_left'][1] - config.EI.points_EI_km[:,1], 
    config.EI.EI_triangles, facecolors=face_colors, vmin = 0, vmax = np.max(config.EI.carrying_cap)*2, cmap="Greens", alpha=1)

    watertriangles = np.zeros([config.EI.N_els])
    watertriangles[config.EI.water_triangle_inds]=1
    #watercmap =  LinearSegmentedColormap.from_list("watercmap",[(0,0,0,0),(127/255,1,212/255 ,1 )],N=2) # AQUAMARINE
    watercmap =  LinearSegmentedColormap.from_list("watercmap",[(0,0,0,0),(3/255,169/255,244/255 ,1 )],N=2) # Indigo blue

    _ = ax.tripcolor(config.EI.points_EI_km[:,0], config.EI.corners['upper_left'][1] - config.EI.points_EI_km[:,1], 
    config.EI.EI_triangles, facecolors=watertriangles, vmin = 0, vmax = 1, cmap=watercmap, alpha=None) 

    #for i in config.EI.water_midpoints:
    #    ax.plot([i[0]], [config.EI.corners['upper_left'][1]- i[1]], 'o',markersize=5, color="blue")
    
    ax.set_aspect('equal')
    divider = make_axes_locatable(plt.gca())
    if save:
        cax = divider.append_axes("right", "5%", pad="3%")
        cb1 = colorbar(treePlot, cax =cax) 
        cb1.set_label_text("Number of Trees in Triangle")
    return fig, ax, divider, treePlot

#  



def plot_single_Penalty():
    return 


#####################################

def plot_statistics(data, folder):
    fig = plt.figure(figsize=(20,6))
    rcParams = {'font.size':20}
    plt.rcParams.update(rcParams)
    #ax1.spines["right"].set_position(('axes',1.0))

    colors=["blue", "k", "green", "red", "orange"]


    ax = []
    ax.append(fig.add_subplot(111))
    for i in range(4):
        ax.append(ax[0].twinx())


    (1e-3 * data.total_population).plot(ax=ax[0],lw=3, color=colors[0])
    ax[0].set_ylabel("Total Population in 1000s")

    excess_mortality = data.total_excessdeaths/data.total_population
    excess_births = data.total_excessbirths/data.total_population

    N=50
    mortality_smoothed = np.convolve(excess_mortality, np.ones((N,))/N, mode='valid')
    time_smoothed = np.convolve(data.time, np.ones((N,))/N, mode='valid')
    ax[1].plot(time_smoothed, mortality_smoothed*100, ":", color=colors[1], lw=3)
    ax[1].set_ylabel("Smoothed (Excess-)Mortality [%]")

    births_smoothed = np.convolve(excess_births, np.ones((N,))/N, mode='valid')
    time_smoothed = np.convolve(data.time, np.ones((N,))/N, mode='valid')
    ax[1].plot(time_smoothed, births_smoothed*100, "--", color=colors[1], lw=3)
    ax[1].set_ylabel("ExcessBirths/Mortality [%]")


    (data.total_trees*1e-6).plot(ax=ax[2],color=colors[2], lw=3)
    ax[2].set_ylabel("Nr of Trees (in mio)")

    if len(data.firesTime)>0:
        fireSize = np.zeros(len(data.time))
        fireSize[data.firesTime- data.isel(time=0).time.values] += data.firesSize
        t0=data.isel(time=0).time.values
        BINwidth=20
        bins = np.arange(t0, data.isel(time=-1).time+1, step=BINwidth )
        binned_fireSize = [1e-3 * np.sum( fireSize[( bins[k]-t0)  :  (bins[k+1]-t0 )]) for k in range(len(bins)-1)]
        ax[3].bar(bins[:-1], binned_fireSize,color=colors[3], width=BINwidth)#, bins=np.arange(data.time[0],data.time[-1]+1, step=20), alpha=1, color=colors[3])
        ax[3].set_ylabel("Fires [1000 Trees /"+str(BINwidth)+" yrs]")

    
    plt.fill_between(data.time, data.happyMeans-data.happyStd, data.happyMeans+data.happyStd, color=colors[4], alpha=0.3)
    data.happyMeans.plot(ax=ax[4],color=colors[4], alpha=1)
    plt.fill_between(data.time, data.Penalty_Mean-data.Penalty_Std, data.Penalty_Mean+data.Penalty_Std, color=colors[4], alpha=0.3)
    data.Penalty_Mean.plot(ax=ax[4], color=colors[4], alpha=1)
    ax[4].set_ylabel("Mean Penalty and happiness")



    for i,a in enumerate(ax):
        #a.spines["right"].set_position(('axes',1.0))
        a.tick_params(axis="y", colors=colors[i])
        a.spines["right"].set_position(("axes", 1+(i)*0.17))
        #a.set_ylabel(["R", "L", "E", "V"][i])
        a.yaxis.label.set_color(colors[i])
        a.set_ylim(0,)
        a.yaxis.set_label_position("right")
        a.yaxis.tick_right()

    ax[3].set_ylim(0,10)

    ax[3].set_zorder(ax[0].get_zorder()) 
    ax[4].set_zorder(ax[3].get_zorder()+1)
    ax[4].patch.set_visible(False)
    ax[1].set_zorder(ax[4].get_zorder()+1)
    ax[1].patch.set_visible(False)
    ax[2].set_zorder(ax[1].get_zorder()+1)
    ax[2].patch.set_visible(False)
    ax[0].set_zorder(ax[2].get_zorder()+1)
    ax[0].patch.set_visible(False)
    fig.tight_layout()
    plt.savefig(folder+"Statistics_timeseries.svg")
    plt.close()
    return 









    ###########################
    ####  OLD   ###############



# def observe_density(t, fig=None, ax=None, save=True, hist=False, folder=config.folder, specific_ag_to_follow=None):# agents, EI, t, agent_type, folder):


#     if ax==None:
#         fig = plt.figure(figsize=(10,6))
#         ax = fig.add_subplot(111)        

#     fig, ax = config.EI.plot_TreeMap_and_hist(ax=ax, fig=fig, hist=hist,folder=folder)
#     #ax.tripcolor(config.EI.points_EI_km[:,0], config.EI.corners['upper_left'][1] - config.EI.points_EI_km[:,1], 
#     #    config.EI.EI_triangles, facecolors=np.array([1 for _ in range(config.EI.N_els)]), vmin = 0, vmax = 1, cmap="Blues", alpha=1)
#     #
#     #plot = ax.tripcolor(config.EI.points_EI_km[:,0], config.EI.corners['upper_left'][1] - config.EI.points_EI_km[:,1], 
#     #  config.EI.EI_triangles, facecolors=config.EI.tree_density, vmin = 0, vmax = np.max(config.EI.carrying_cap), cmap="Greens", alpha=1)
#     ##cbaxes = fig.add_axes([0.9, 0.0, 0.03, 1.0]) 
#     #fig.colorbar(plot)#, cax = cbaxes)  
#     ax = plot_agents_on_top(ax, t,ncdf=False, specific_ag_to_follow=specific_ag_to_follow)

#     if save:
#         ax.set_title("Time "+str(t))
#         #fig.tight_layout()
#         plt.savefig(config.folder+"map_time"+str(t)+".png")
#         plt.close()
#     return fig,ax