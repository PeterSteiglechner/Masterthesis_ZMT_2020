#!/usr/bin/env python

# PETEr STEIGLECHNER, 9.3.2020, 
# 
''' This File contains the 
 - observe function where treees are agents (for HopVes_i Base and _HHet)
 - observe_density function where trees are just a number associated with a triangle on Easter Island (HopVes_i_spatExt, +_hh_ext)
 - follow_ag
 '''

import config

import matplotlib as mpl 
import numpy as np
import matplotlib.pyplot as plt
font = {'font.size':15}
plt.rcParams.update(font)
import matplotlib.colors as colors

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import sys, os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

widthcbar ="5%"
distwbar="15%"
initial_distwbar = "2%"


   
def observe(t, fig=None, ax=None, specific_ag_to_follow=None, save=True, data = None, ncdf=False, folder=None, posSize_small=False, legend=False, cbar=[True, True, True], cbarax=None):
    global distwbar, initial_distwbar
    if posSize_small:
        distwbar="5%"
        initial_distwbar = "5%"
    else:
        distwbar="15%"
        initial_distwbar="2%"
    if fig==None:
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(1,1,1,fc='gainsboro')
    if folder==None:
        folder=config.folder
    fig, ax, divider, _, cmapTree =   plot_TreeMap_only(fig, ax, t, data=data, ncdf=ncdf, save = cbar[0], cbarax=cbarax)
    ax, divider, _, cmapAgric  = plot_agricultureSites_onTop(ax, divider, t, data=data, ncdf=ncdf, save =cbar[1], cbarax=cbarax)
    if len(config.agents)>0 or (ncdf==True):
        ax,divider, _,cmapPop = plot_agents_on_top(ax, divider,t, ncdf=ncdf, data=data, specific_ag_to_follow=specific_ag_to_follow, save=cbar[2], cbarax=cbarax)
    else:
        cmapPop=0
    ax.set_title(str(t)+r"$\,$A.D.", x=0.01, y=0.9,fontweight="bold", loc='left')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    
    scalebar = AnchoredSizeBar(ax.transData,
                            5, '5 km', 'lower center', 
                            pad=0.4,
                            color='black',
                            frameon=False,
                            size_vertical=0.2,
                            fontproperties= fm.FontProperties(size=20))#*(1+posSize_small*1)))
    ax.add_artist(scalebar)

    lake = Patch(label="Lake", facecolor=config.lakeColor)
    ax.legend(handles = [lake], loc="lower right", frameon=False, fontsize=20)
    if save==True:
        if ncdf==False:
            ending = ".png"
        else:
            ending=".pdf"
        plt.savefig(folder+"map_time"+str(t)+ending, bbox_inches="tight")
        plt.close()

    return fig, ax, [cmapTree, cmapAgric, cmapPop] 



def follow_ag(x,y,mr, tr, ar):
    ''' plot circle around ag0 with resource and move search radius'''
    move_circle = plt.Circle([x, config.EI.corners['upper_left'][1]-y], 
        radius = mr, clip_on=True, color='black', alpha=0.1)
    tree_circle = plt.Circle([x, config.EI.corners['upper_left'][1]-y], 
        radius = tr, clip_on=True, color='black', alpha=0.2)
    agric_circle = plt.Circle([x, config.EI.corners['upper_left'][1]-y], 
        radius = ar, clip_on=True, color='black', alpha=0.3)
    return move_circle, tree_circle, agric_circle






######################################################################
###########        PLOT FUNCTIONS LATER                 ##############
######################################################################
def plot_agents_on_top(ax, divider, t, ncdf=False, data=None, specific_ag_to_follow=None, posSize_small=False, save=True, cbarax=None):
    if ncdf==False:
        agent_points_np = np.array([[ag.x, ag.y, ag.pop] for ag in config.agents])# subtracting ag.y from the upper left corner gives a turned picture (note matrix [0][0] is upper left)
        agent_size = agent_points_np[:,2]
    else:
        agent_points_np = data.PosAgents.sel(time=t).to_series().dropna()
        agent_size = data.SizeAgents.sel(time=t).to_series().dropna()
        agent_points_np = np.array([agent_points_np[::2], agent_points_np[1::2]]).T
        agent_treePref = data.TreePrefAgents.sel(time=t).to_series().dropna()

    #print(agent_treePref.shape)
    if ncdf == False:
        #color = [((1-ag.treePref), 0, 1,1) for ag in config.agents]
        color =[(ag.T_Pref_i) for ag in config.agents]
    else:
        #color = [((1-agent_treePref[k]), 0, 1,1) for k in agent_treePref.index]
        color = [(agent_treePref[k]) for k in agent_treePref.index]

    colorscmap = [(1,0,1,1), (0,0,1,1)]
    cmapPop = LinearSegmentedColormap.from_list("bluepurple",colorscmap[:], N=256)

    if len(agent_points_np)>0:
        factor = 0.02 if posSize_small else 0.2
        plot = ax.scatter(agent_points_np[:,0], config.EI.corners['upper_left'][1] - agent_points_np[:,1], s=agent_size*factor, c=color, cmap = cmapPop, vmin=0.2, vmax=0.8)#'purple')

        if ncdf==False:
            if not specific_ag_to_follow==None:
                #specific_ag_to_follow = config.agents[0]
                a = specific_ag_to_follow
                move_circle, tree_circle, agric_cirle = follow_ag(a.x, a.y, a.moving_radius, a.tree_search_radius, a.agriculture_radius)
                if not a.moving_radius==100: ax.add_artist(move_circle)
                ax.add_artist(tree_circle)
                ax.add_artist(agric_cirle)
        if save:
            if cbarax==None:
                cax2 = divider.append_axes("right", widthcbar, pad=distwbar)
            else:
                cax2=cbarax            
            cb2 = colorbar(plot, cax =cax2, ticks=[0.2,0.4,0.6,0.8])   
            cb2.ax.set_yticklabels([str(k) for k in [0.2,0.4,0.6,0.8]])
            cb2.set_label_text("Tree Preference of Agents")
    return ax, divider, plot, cmapPop



   

def plot_agricultureSites_onTop(ax, divider, t, data=None, ncdf=False, save=True, cbarax=None):
    if ncdf:
        face_colors = data.F_ct.sel(time=t)* 1/(config.EI.A_c * config.km2_to_acre)  #(config.EI.nr_highqualitysites+config.EI.nr_lowqualitysites).clip(min=0.1)
    else:
        face_colors = config.EI.A_F_c* 1/(config.EI.A_c * config.km2_to_acre)  #(config.EI.nr_highqualitysites+config.EI.nr_lowqualitysites).clip(min=0.1)
    if any(face_colors>1):
        print("Error in plot agricultureSites")
    #colors = [(1,0,0,0), (1,0,0,1)]
    #customCmap = LinearSegmentedColormap.from_list("agriccmap",colors,N=256)
    oranges = plt.get_cmap("Oranges")(np.linspace(0, 0.5, int(256*1)))
    oranges[:256,3] = np.linspace(0.0,0.9,256)
    cmap = LinearSegmentedColormap.from_list("orangeshalf",oranges[:256])

    #plot_agricultureSites_onTop(ax=ax, fig=fig)    
    AgricPlot = ax.tripcolor(config.EI.points_EI_km[:,0], config.EI.corners['upper_left'][1] - config.EI.points_EI_km[:,1], config.EI.EI_triangles, facecolors=face_colors, cmap=cmap, alpha=None, vmax=1, vmin=0)

    if save:
        if cbarax==None:
            cax2 = divider.append_axes("right", widthcbar, pad=distwbar)
        else:
            cax2=cbarax
        cb2 = colorbar(AgricPlot, cax =cax2)     
        cb2.set_label_text("Fraction of farmed land")# [${\rm acre/acre}$]")
    return ax, divider, AgricPlot, (cmap, oranges[255])


def plot_TreeMap_only(fig, ax, t, data=None, ncdf=False, save=True, cbarax=None):
    if ncdf:
        face_colors =data.T_ct.sel(time=t)/(config.EI.A_c*1e6)
        face_colors[(data.T_ct.sel(time=800)>0)*(face_colors==0)] = -np.spacing(0.0)
        vmax = np.max(data.T_ct.sel(time=800)/(config.EI.A_c*1e6))
    else:
        face_colors = config.EI.T_c/(config.EI.A_c*1e6)
        face_colors[(config.EI.carrying_cap>0)*(face_colors==0)] = -np.spacing(0.0)
        vmax = np.max(config.EI.carrying_cap/(config.EI.A_c*1e6))

    green = plt.get_cmap("Greens")(np.linspace(0, 1, int(256*1.5)))
    cmap = LinearSegmentedColormap.from_list("greenhalf",green[:256])
    #cmap.set_under("brown")
    treePlot = ax.tripcolor(config.EI.points_EI_km[:,0], config.EI.corners['upper_left'][1] - config.EI.points_EI_km[:,1], 
    config.EI.EI_triangles, facecolors=face_colors, vmin = 0, vmax=vmax, cmap=cmap, alpha=1)

    watertriangles = np.zeros([config.EI.N_c])
    if ncdf:
        d1 = [data.drought_RanoRaraku_1_start, data.drought_RanoRaraku_1_end]
        d2 = [data.drought_RanoRaraku_2_start, data.drought_RanoRaraku_2_end]
    else:
        d1 = config.droughts_RanoRaraku[0]
        d2 = config.droughts_RanoRaraku[1]
    if (t>d1[0] and t<d1[1]) or (t>d2[0] and t< d2[1]):
        watertriangles[config.EI.water_triangle_inds_RarakuDrought]=1
    else:
        watertriangles[config.EI.water_triangle_inds_NoDrought]=1
    #watercmap =  LinearSegmentedColormap.from_list("watercmap",[(0,0,0,0),(127/255,1,212/255 ,1 )],N=2) # AQUAMARINE
    watercmap =  LinearSegmentedColormap.from_list("watercmap",[(0,0,0,0),config.lakeColor],N=2) # Indigo blue

    _ = ax.tripcolor(config.EI.points_EI_km[:,0], config.EI.corners['upper_left'][1] - config.EI.points_EI_km[:,1], 
    config.EI.EI_triangles, facecolors=watertriangles, vmin = 0, vmax = 1, cmap=watercmap, alpha=None) 

    #for i in config.EI.water_midpoints:
    #    ax.plot([i[0]], [config.EI.corners['upper_left'][1]- i[1]], 'o',markersize=5, color="blue")
    
    ax.set_aspect('equal')
    divider = make_axes_locatable(plt.gca())
    if save:
        if cbarax==None:
            cax = divider.append_axes("right", widthcbar, pad=initial_distwbar)
        else:
            cax=cbarax
        cb1 = colorbar(treePlot, cax =cax)
        cb1.ax.set_yticks(np.arange(0,1/100*np.ceil(100*np.max(face_colors)), step=0.03) )
        cb1.ax.set_yticklabels([str(int(k*100)) for k in cb1.ax.get_yticks()])
        cb1.set_label_text(r"Tree Density [$\rm{Trees/100\,m^2}$] ")
    return fig, ax, divider, treePlot, (cmap, green[255])

#  








###########################################################
######    MOVING PROB              ########################
###########################################################


def plot_movingProb(P, folder, data):
    # [P['AG_total_penalties'],
    #             P['AG_water_penalties'],
    #             P['AG_tree_penalty'],
    #             P['AG_pop_density_penalty'],
    #             P['AG_agriculture_penalty'],
    #             P['AG_map_penalty'],
    #             P['AG_nosettlement_zones'],
    #             P['AG_MovingProb'],
    #             P['AG_Pos'],
    #             P['AG_NewPos'],
    #             P['AG_InitialTriangle'],
    #             _,#NewPosTriangle,
    #             P['AG_MovingRadius'],
    #             P['AG_TPref'],
    #             P['AG_pop'],
    #             _,#AG_H
    #             P['t'],
    #             AG_index] = [ToSavePenalties[key] for key in ToSavePenalties.keys()]
    
    plt.rcParams.update({"font.size":15})
    fig = plt.figure(figsize=(9*3,6*3))
    #gs = fig.add_gridspec(3,4,width_ratios=[18,18,18,1])

    ax = fig.add_subplot(3,3,1,fc='gainsboro')
    #rcParams={'font.size':10}
    #plt.rcParams.update(rcParams)

    fig, ax, _= observe(P['t'], fig=fig, ax=ax, specific_ag_to_follow=None, data = data, ncdf=True, save=False, cbar=[False, False,False]) #(t, fig=fig, ax=ax, save=False, hist=False, folder="", specific_ag_to_follow=ag)
    #fig,ax = config.EI.plot_agricultureSites(ax=ax, fig=fig, save=False, CrossesOrFacecolor="Crosses")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    ax.set_title("Agent "+str(P['AG_index']))

    #ax.scatter(P['AG_Pos'][0], config.EI.corners['upper_left'][1] - P['AG_Pos'][0], s=P['AG_pop']*factor, c=P["AG_TPref"], cmap = cmapPop, vmin=0.2, vmax=0.8)#'purple')
    #ax.scatter(P['AG_Pos'][0], config.EI.corners['upper_left'][1] - P['AG_Pos'][0], s=P['AG_pop']*factor, c="red", cmap = cmapPop, vmin=0.2, vmax=0.8)#'purple')
    #ax.annotate()
    ax.annotate("", xy=[P["AG_NewPos"][0], config.EI.corners['upper_left'][1] - P["AG_NewPos"][1]],
                 xytext=[P["AG_Pos"][0], config.EI.corners['upper_left'][1] - P["AG_Pos"][1]],
                 arrowprops=dict(arrowstyle="->"))

    move_circle, tree_circle, agric_circle = follow_ag(P['AG_Pos'][0], P['AG_Pos'][1], P['AG_MovingRadius'], config.EI.r_T, config.EI.r_F)
    if P['AG_MovingRadius'] <100: ax.add_artist(move_circle)
    ax.add_artist(tree_circle)
    ax.add_artist(agric_circle)

    ax.set_xlim(max(P['AG_Pos'][0]-P['AG_MovingRadius']*1.1, 0), min(P['AG_Pos'][0]+P['AG_MovingRadius']*1.1, config.EI.corners["upper_right"][0]))
    ax.set_ylim(max(config.EI.corners['upper_left'][1]-P['AG_Pos'][1]-P['AG_MovingRadius']*1.1,0), min(config.EI.corners['upper_left'][1]-P['AG_Pos'][1]+P['AG_MovingRadius']*1.1, config.EI.corners["upper_right"][1]))
       
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
    if P['AG_MovingRadius']<100:
        mv_inds = config.EI.C_Mlater_c[P['AG_InitialTriangle']]
    else:
        mv_inds = np.arange(config.EI.N_c)
    #penalties = [config.AG0_total_penalties, config.AG0_MovingProb, config.AG0_water_penalties, config.AG0_tree_penalty, config.AG0_agriculture_penalty, config.AG0_pop_density_penalty, config.AG0_map_penalty, config.AG0_nosettlement_zones]
    penalties = [P['AG_total_penalties'], P['AG_MovingProb'], P['AG_water_penalties'], P['AG_tree_penalty'], P['AG_agriculture_penalty'], P['AG_pop_density_penalty'], P['AG_map_penalty'], P['AG_nosettlement_zones']]
    penalties_key = ["total_penalties", "MovingProb","water_penalties", "tree_penalty", "agriculture_penalty", "pop_density_penalty", "Map_Penalty","Allowed_settlements"]
    plot=[None for _ in range(10)]
    for k, penalty,key in zip([2,3,4,5,6,7,8,9], penalties, penalties_key):
        ax2  = fig.add_subplot(3,3,k, fc="aquamarine")
        #ax2.tripcolor(config.EI.points_EI_km[:,0], config.EI.corners['upper_left'][1] - config.EI.points_EI_km[:,1], 
        #        config.EI.EI_triangles, facecolors=np.array([1 for _ in range(config.EI.N_els)]), vmin = 0, vmax = 1, cmap="binary", alpha=1)

        ax2.set_xlim(max(P['AG_Pos'][0]-P['AG_MovingRadius']*1.1, 0), min(P['AG_Pos'][0]+P['AG_MovingRadius']*1.1, config.EI.corners["upper_right"][0]))
        ax2.set_ylim(max(0,config.EI.corners['upper_left'][1]-P['AG_Pos'][1]-P['AG_MovingRadius']*1.1), min(config.EI.corners['upper_left'][1]-P['AG_Pos'][1]+P['AG_MovingRadius']*1.1, config.EI.corners["upper_right"][1]))
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

            colors = np.array([0 for i in range(config.EI.N_c)])
            #colors[config.AG0_mv_inds] = [liste[mpl.colors.to_hex(penalty[:,i], keep_alpha=True)] for i in range(penalty.shape[1])]
            colors[mv_inds] = penalty#[1 if np.sum(penalty[:,i])==4 else 0 for i in range(penalty.shape[1])]
            plot  = ax2.tripcolor(config.EI.points_EI_km[:,0], config.EI.corners['upper_left'][1] - config.EI.points_EI_km[:,1], 
                config.EI.EI_triangles, facecolors=colors, vmin = 0, cmap='binary_r',alpha=1)
        else:
            prob = np.zeros([config.EI.N_c])-np.spacing(0.0)
            prob[mv_inds] = penalty

            if k==3:
                #cmap="hot_r"
                prob*=3
                cmap = plt.get_cmap('magma_r')
                cmap.set_under('gray') 
                vmin = 0.0
                vmax=np.max(prob).clip(max=1)
            elif k ==2:
                cmap = plt.get_cmap("Reds")
                prob[np.where(prob>1)]=-1.0
                cmap.set_under('gray') 
                vmax=np.max(prob).clip(min=0.01, max=1)
                vmin= 0.0                
            else:
                cmap = plt.get_cmap("Reds")
                cmap.set_under('gray') 
                vmin = 0.0
                vmax=1
                if k==6:
                    vmin = -1.0# np.min(penalty)
                    prob = np.zeros([config.EI.N_c])-2.0-np.spacing(0.0)
                    prob[mv_inds] = penalty
                    vmax = 1
                    colorsneg = plt.get_cmap("Blues_r")(np.linspace( 0.0,1, 128))
                    colorspos = plt.get_cmap("Reds")(np.linspace(0, 1.0, 128))
                    colorsstack = np.vstack((colorsneg, colorspos))
                    cmap = LinearSegmentedColormap.from_list('my_colormap', colorsstack)
                    cmap.set_under('gray') 

            plot[k]  = ax2.tripcolor(config.EI.points_EI_km[:,0], config.EI.corners['upper_left'][1] - config.EI.points_EI_km[:,1], 
                config.EI.EI_triangles, facecolors=prob,vmin=vmin,vmax=vmax, cmap=cmap, alpha=1)


                

        #if not all(config.AG0_mv_inds==P['AG_MovingRadius']):
        #    print("Error in Plot Penalties")
        ax2.plot(P['AG_NewPos'][0], config.EI.corners['upper_left'][1]-P['AG_NewPos'][1], "o", markersize = int(P['AG_pop']/3),color=((1-P['AG_TPref']), 0, 1,1), fillstyle="none")
        ax2.plot(P['AG_Pos'][0], config.EI.corners['upper_left'][1]-P['AG_Pos'][1], "o", markersize = int(P['AG_pop']/3) ,color=((1-P['AG_TPref']), 0, 1,1))
        
    
        
        ax2.set_title(key)
        #print(np.min(prob),np.max(prob))
        #move_circle, tree_circle, agric_cirle = follow_ag(ag)
        #ax2.add_artist(move_circle)
        #ax2.add_artist(tree_circle)
        #ax2.add_artist(agric_cirle)
        #divider = make_axes_locatable(plt.gca())
        #cax = divider.append_axes("right", "5%", pad="3%")
        #plt.colorbar(plot[-1], cax =cax) 

    #fig.tight_layout()
    #fig.colorbar(p)
    plt.savefig(folder+"Penalties_AG"+str(P['AG_index'])+"_t="+str(P['t'])+".svg", bbox_inches='tight')























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


    (1e-3 * data.pop_ct.sum(dim="triangles")).plot(ax=ax[0],lw=3, color=colors[0])
    ax[0].set_ylabel("Total Population in 1000s")

    excess_mortality = data.total_excessdeaths/data.pop_ct.sum(dim="triangles")
    excess_births = data.total_excessbirths/data.pop_ct.sum(dim="triangles")

    N=50
    mortality_smoothed = np.convolve(excess_mortality, np.ones((N,))/N, mode='valid')
    time_smoothed = np.convolve(data.time, np.ones((N,))/N, mode='valid')
    ax[1].plot(time_smoothed, mortality_smoothed*100, ":", color=colors[1], lw=3)
    #ax[1].set_ylabel("Smoothed (Excess-)Mortality [%]")

    births_smoothed = np.convolve(excess_births, np.ones((N,))/N, mode='valid')
    time_smoothed = np.convolve(data.time, np.ones((N,))/N, mode='valid')
    ax[1].plot(time_smoothed, births_smoothed*100, "--", color=colors[1], lw=3)
    ax[1].set_ylabel("Excess Births(dashed)/Mortality(dotted) [%]")


    (data.T_ct.sum(dim='triangles')*1e-6).plot(ax=ax[2],color=colors[2], lw=3)
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
    data.Penalty_Mean.plot(ax=ax[4],linestyle="--", color=colors[4], alpha=1)
    ax[4].set_ylabel("Mean Penalty (dashed) and happiness")



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


















def plot_F_PI_Map():
        #distwbar="15%"
        initial_distwbar="2%"
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(1,1,1,fc='gainsboro')
        #ax, divider, _, _  = plot_agricultureSites_onTop(ax, divider, t, data=data, ncdf=ncdf, save =cbar[1], cbarax=cbarax)
        face_colors = config.EI.F_pi_c #(config.EI.nr_highqualitysites+config.EI.nr_lowqualitysites).clip(min=0.1)
        face_colors[face_colors==0]=-1.
        oranges = plt.get_cmap("Wistia", 11)#(np.linspace(0, 1, int(10*1)))
        #oranges[:256,3] = np.linspace(0.0,0.9,256)
        #cmap = LinearSegmentedColormap.from_list("orangeshalf",oranges)
        cmap = oranges
        cmap.set_under("white")
        #plot_agricultureSites_onTop(ax=ax, fig=fig)    
        AgricPlot = ax.tripcolor(config.EI.points_EI_km[:,0], config.EI.corners['upper_left'][1] - config.EI.points_EI_km[:,1], config.EI.EI_triangles, facecolors=face_colors, cmap=cmap, alpha=None, vmax=1, vmin=0)
        
        watertriangles = np.zeros([config.EI.N_c])
        watertriangles[config.EI.water_triangle_inds_NoDrought]=1
        watercmap =  LinearSegmentedColormap.from_list("watercmap",[(0,0,0,0),config.lakeColor],N=2) # Indigo blue
        _ = ax.tripcolor(config.EI.points_EI_km[:,0], config.EI.corners['upper_left'][1] - config.EI.points_EI_km[:,1], 
        config.EI.EI_triangles, facecolors=watertriangles, vmin = 0.0, vmax = 1, cmap=watercmap, alpha=None) 
        ax.set_aspect('equal')
        divider = make_axes_locatable(plt.gca())
        cax2 = divider.append_axes("right", widthcbar, pad=initial_distwbar)
        cb2 = colorbar(AgricPlot, cax =cax2)#, extend='min')     
        cb2.set_label_text("Farming Productivity Index \n"+r"$F_{\rm PI}(c, t=t_{\rm arrival})$")# [${\rm acre/acre}$]")
        #ax.set_title(str(t)+r"$\,$A.D.", x=0.01, y=0.9,fontweight="bold", loc='left')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        scalebar = AnchoredSizeBar(ax.transData,
                            5, '5 km', 'lower center', 
                            pad=0.4,
                            color='black',
                            frameon=False,
                            size_vertical=0.2,
                            fontproperties= fm.FontProperties(size=20))#*(1+posSize_small*1)))
        ax.add_artist(scalebar)
        #waterbox = plt.Rectangle((0.75,0.8), 0.1,0.06, visible=True, label="Lakes", fc=config.lakeColor, transform = ax.transAxes)   
        #ax.add_artist(waterbox)
        #plt.text(0.95, 0.05, "Lakes", fontsize=15*(1+posSize_small*1), transform = ax.transAxes)
        
        lake = Patch(label="Lake", facecolor=config.lakeColor)
        ax.legend(handles = [lake], loc="lower right", frameon=False, fontsize=20)

        
        plt.savefig("Map/Plot_F_PI_c.pdf")

        return #ax, divider, AgricPlot, (cmap, oranges[255])






def plot_Penalty_Map(which, label, name):
        #distwbar="15%"
        initial_distwbar="2%"
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(1,1,1,fc='gainsboro')
        #ax, divider, _, _  = plot_agricultureSites_onTop(ax, divider, t, data=data, ncdf=ncdf, save =cbar[1], cbarax=cbarax)
        face_colors = which #(config.EI.nr_highqualitysites+config.EI.nr_lowqualitysites).clip(min=0.1)
        #face_colors[face_colors==0]=-1.
        #oranges = plt.get_cmap("Wistia", 11)#(np.linspace(0, 1, int(10*1)))
        #oranges[:256,3] = np.linspace(0.0,0.9,256)
        #cmap = LinearSegmentedColormap.from_list("orangeshalf",oranges)
        #cmap = oranges
        cmap = plt.get_cmap("Reds")
        cmap.set_under('gray') 
        #vmin = 0.0
        #vmax=1        #plot_agricultureSites_onTop(ax=ax, fig=fig)    
        PGPlot = ax.tripcolor(config.EI.points_EI_km[:,0], config.EI.corners['upper_left'][1] - config.EI.points_EI_km[:,1], config.EI.EI_triangles, facecolors=face_colors, cmap=cmap, alpha=None, vmax=1, vmin=0)
        
        watertriangles = np.zeros([config.EI.N_c])
        watertriangles[config.EI.water_triangle_inds_NoDrought]=1
        watercmap =  LinearSegmentedColormap.from_list("watercmap",[(0,0,0,0),config.lakeColor],N=2) # Indigo blue
        _ = ax.tripcolor(config.EI.points_EI_km[:,0], config.EI.corners['upper_left'][1] - config.EI.points_EI_km[:,1], 
        config.EI.EI_triangles, facecolors=watertriangles, vmin = 0.0, vmax = 1, cmap=watercmap, alpha=None) 
        
        ax.set_aspect('equal')
        divider = make_axes_locatable(plt.gca())
        cax2 = divider.append_axes("right", widthcbar, pad=initial_distwbar)
        cb2 = colorbar(PGPlot, cax =cax2)#, extend='min')     
        cb2.set_label_text(label)# [${\rm acre/acre}$]")
        #ax.set_title(str(t)+r"$\,$A.D.", x=0.01, y=0.9,fontweight="bold", loc='left')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        scalebar = AnchoredSizeBar(ax.transData,
                            5, '5 km', 'lower center', 
                            pad=0.4,
                            color='black',
                            frameon=False,
                            size_vertical=0.2,
                            fontproperties= fm.FontProperties(size=20))#*(1+posSize_small*1)))
        ax.add_artist(scalebar)
        #waterbox = plt.Rectangle((0.75,0.8), 0.1,0.06, visible=True, label="Lakes", fc=config.lakeColor, transform = ax.transAxes)   
        #ax.add_artist(waterbox)
        #plt.text(0.95, 0.05, "Lakes", fontsize=15*(1+posSize_small*1), transform = ax.transAxes)
        
        lake = Patch(label="Lake", facecolor=config.lakeColor)
        ax.legend(handles = [lake], loc="lower right", frameon=False, fontsize=20)

        
        plt.savefig("Map/Plot_"+name+".pdf")

        return #ax, divider, AgricPlot, (cmap, oranges[255])













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

if __name__=="__main__":
    import numpy as np 
    import xarray as xr 
    import matplotlib.pyplot as plt 
    import observefunc as obs
    plt.rcParams.update({"font.size":18})

    #folder="/home/peter/EasterIslands/Code/Full_Model/Figs_May11_grid50/FullModel_grid50_repr7e-03_mv100_noRegrowth_highFix_seed101/"
    folder="/home/peter/EasterIslands/Code/Full_Model/Figs_May18/FullModel_grid50_gH11e+00_noRegrowth_highFix_seed5/"
    data = xr.open_dataset(folder+"Statistics.ncdf")

    import pickle
    import config
    from pathlib import Path   # for creating a new directory
    filename = "Map/EI_grid"+str(data.gridpoints_y)+"_rad"+str(data.r_T)+"+"+str(data.r_F)+"+"+str(data.r_M_later)
    if Path(filename).is_file():
        with open(filename, "rb") as EIfile:
            config.EI = pickle.load(EIfile)


    with open(folder+"Penalties_AG500_t=1638", "rb") as PenaltyFile:
        P = pickle.load(PenaltyFile)

    plot_movingProb(P, folder, data)