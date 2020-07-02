# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import xarray as xr 
import matplotlib.pyplot as plt 
import numpy as np 
import matplotlib as mpl 
from matplotlib.ticker import MaxNLocator
rcParams = {'font.size':20}
plt.rcParams.update(rcParams)


# %%
folder_std = "/home/peter/EasterIslands/Runs_22May/Standard/"
folders = [folder_std+"FullModel_grid50_gH17e-3_noRegrowth_highFix_linear_NormPop_alphaStd_seed"+str(seed)+"/" for seed in np.arange(1,16)]
stds = [xr.open_dataset(folder+"Statistics.ncdf") for folder in folders]
label_Standard1 = "Standard"
label_Standard = "Standard: High N fixation\n(without tree regrowth)"


# %%
folder_delayed = "/home/peter/EasterIslands/Runs_22May/delayed/"
folders = [folder_delayed+"FullModel_grid50_gH17e-3_noRegrowth_highFix_delayed_NormPop_alphaStd_seed"+str(seed)+"/" for seed in np.arange(1,16)]
delayed = [xr.open_dataset(folder+"Statistics.ncdf") for folder in folders]
label_delayed = r"Delayed $T_{\rm Pref}$-Response"


# %%
folder_careful= "/home/peter/EasterIslands/Runs_22May/careful/"
folders = [folder_careful+"FullModel_grid50_gH17e-3_noRegrowth_highFix_careful_NormPop_alphaStd_seed"+str(seed)+"/" for seed in np.arange(1,16)]
careful = [xr.open_dataset(folder+"Statistics.ncdf") for folder in folders]
label_careful = r"Early $T_{\rm Pref}$-Response"


# %%
folder_logistic = "/home/peter/EasterIslands/Runs_22May/logistic/"
folders = [folder_logistic+"FullModel_grid50_gH17e-3_noRegrowth_highFix_logistic_NormPop_alphaStd_seed"+str(seed)+"/" for seed in np.arange(1,16)]
logistic = [xr.open_dataset(folder+"Statistics.ncdf") for folder in folders]
label_logistic = r"Logistic $T_{\rm Pref}$-Response"


# %%
folder_with = "/home/peter/EasterIslands/Runs_22May/with/"
folders = [folder_with+"FullModel_grid50_gH17e-3_withRegrowth_highFix_linear_NormPop_alphaStd_seed"+str(seed)+"/" for seed in np.arange(1,16)]
withs = [xr.open_dataset(folder+"Statistics.ncdf") for folder in folders]
label_with = "High N fixation\n(with tree regrowth)"


# %%
folder_low = "/home/peter/EasterIslands/Runs_22May/low/"
folders = [folder_low+"FullModel_grid50_gH17e-3_noRegrowth_lowFix_linear_NormPop_alphaStd_seed"+str(seed)+"/" for seed in np.arange(1,16)]
lows = [xr.open_dataset(folder+"Statistics.ncdf") for folder in folders]
label_low = "Low N fixation\n(without tree regrowth)"


# %%
folder_lowwith = "/home/peter/EasterIslands/Runs_22May/lowwith/"
folders = [folder_lowwith+"FullModel_grid50_gH17e-3_withRegrowth_lowFix_linear_NormPop_alphaStd_seed"+str(seed)+"/" for seed in np.arange(1,16)]
lowwiths = [xr.open_dataset(folder+"Statistics.ncdf") for folder in folders]
label_lowwith = "Low N fixation\n(with tree regrowth)"


# %%



# %%


def plot_stats(ax, title):
    datavals = [1e-3 * datas[k].pop_ct.sum(dim="triangles") for k in range(len(datas))]
    mean = np.mean(datavals, axis=0)
    std = np.std(datavals, axis=0)
    ind = 0; label = r"$\mathbf{pop}(t)$  [$10^3 \, {\rm People}$]  "
    for k in range(len(datas)):
        ax[ind].plot(datas[0].time,  datavals[k], alpha = 0.2, color=colors[ind])
    ax[ind].fill_between(datas[0].time, mean-std,  mean +std, color=colors[ind], alpha=0.3)
    a = ax[ind].plot(datas[0].time, mean, color=colors[ind], lw=3, label=label)#, color=colors[0])
    ax[ind].set_ylabel(label)
    ylimhigh[ind] = 24


    datavals = [(datas[k].T_ct.sum(dim='triangles')*1e-6) for k in range(len(datas))]
    mean = np.mean(datavals, axis=0)
    std = np.std(datavals, axis=0)
    ind = 1; label = r"$\mathbf{T}(t)$  [$10^6\, {\rm Trees}]$"
    for k in range(len(datas)):
        ax[ind].plot(datas[0].time,  datavals[k], alpha = 0.2, color=colors[ind])
    ax[ind].fill_between(datas[0].time, mean-std,  mean +std, color=colors[ind], alpha=0.3)
    b = ax[ind].plot(datas[0].time, mean, color=colors[ind], lw=3, label=label)#, color=colors[0])
    ax[ind].set_ylabel(label)
    ylimhigh[ind] = 16


    datavals=[]
    for j in range(0,len(datas)):
        fireSize = np.zeros(len(datas[0].time))
        t0=datas[0].isel(time=0).time.values
        fireSize[datas[j].firesTime- t0] += datas[j].firesSize
        BINwidth=20
        bins = np.arange(t0, datas[0].isel(time=-1).time+1, step=BINwidth )
        binned_fireSize = [1e-3 * np.sum( fireSize[( bins[k]-t0)  :  (bins[k+1]-t0 )]) for k in range(len(bins[:-1]))]
        #ax[3].bar(bins[:-1], binned_fireSize,color=colors[3], width=BINwidth)#, bins=np.arange(data.time[0],data.time[-1]+1, step=20), alpha=1, color=colors[3])
        #ax[3].set_ylabel("Fires [1000 Trees /"+str(BINwidth)+" yrs]")
        #plt.plot(bins[:-1], binned_fireSize, lw=1, color=colors[3])
        datavals.append(binned_fireSize)
    mean = np.mean(datavals, axis=0)
    std = np.std(datavals, axis=0)
    ind = 2; label = r"Burnt Trees  [$\frac{10^3\, {\rm Trees}}{20\, {\rm yrs}}$]"
    for k in range(len(datas)):
        ax[ind].plot(bins[:-1],  datavals[k], alpha = 0.2, color=colors[ind])
    ax[ind].fill_between(bins[:-1], mean-std,  mean +std, color=colors[ind], alpha=0.3)
    c = ax[ind].plot(bins[:-1], mean, color=colors[ind], lw=3, label=label)#, color=colors[0])
    ax[ind].set_ylabel(label)
    ylimhigh[ind] = 14

    for i,a in enumerate(ax):
        #a.spines["right"].set_position(('axes',1.0))
        a.tick_params(axis="y", colors=colors[i])
        a.spines["right"].set_position(("axes", 1.0+(i)*0.1))
        #a.set_ylabel(["R", "L", "E", "V"][i])
        a.yaxis.label.set_color(colors[i])
        a.set_ylim(0,ylimhigh[i])
        a.yaxis.set_label_position("right")
        a.yaxis.tick_right()
        a.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax[0].set_xlabel(r"Year  [${\rm A.D.}$]")
    ax[0].set_title(title, x=0.01, y=0.8,fontweight="bold", loc='left')
    return 

#for datas, label, folder in zip([stds, careful, delayed, logistic], [label_Standard, label_careful, label_delayed, label_logistic], [...]):
#    #datas = careful
#    plot_stats(ax, label)


# %%



# %%
for datas, label, folder in zip([stds, withs, lowwiths, lows], [label_Standard, label_with, label_lowwith, label_low], [folder_std, folder_with, folder_lowwith, folder_low]):
#for datas, label, folder in zip([stds, delayed, careful, logistic], [label_Standard1, label_delayed, label_careful, label_logistic], [folder_std, folder_delayed, folder_careful, folder_logistic]):
    fig =plt.figure(figsize=(16,9))
    colors=["blue", "green", "red"]#, "orange"]
    ax = []
    ax.append(fig.add_subplot(111))
    ax[0].set_xlim(stds[0].time[0], stds[0].time[-1])
    for i in range(2):
        ax.append(ax[0].twinx())
    ylimhigh=[0 for i in range(3)]
    #datas = careful
    plot_stats(ax, label)

    #plt.legend([a,c,b])
    plt.savefig(folder+"EnsembleStatistics.pdf", bbox_inches="tight")
    print(folder)


# %%



# %%
import numpy as np 
import xarray as xr 
import matplotlib.pyplot as plt 
import observefunc as obs
plt.rcParams.update({"font.size":20})

#folder="/home/peter/EasterIslands/Code/Full_Model/Figs_May11_grid50/FullModel_grid50_repr7e-03_mv100_noRegrowth_highFix_seed101/"
import pickle
import config
from pathlib import Path   # for creating a new directory
datas = stds
filename = "Map/EI_grid"+str(datas[0].gridpoints_y)+"_rad"+str(datas[0].r_T)+"+"+str(datas[0].r_F)+"+"+str(datas[0].r_M_later)
if Path(filename).is_file():
    with open(filename, "rb") as EIfile:
        config.EI = pickle.load(EIfile)


# %%
import observefunc as obs
for datas, label, folder in zip([stds, delayed, careful, logistic], [label_Standard1, label_delayed, label_careful, label_logistic], [folder_std, folder_delayed, folder_careful, folder_logistic]):
    _ = obs.observe(1000, fig=None, ax=None, specific_ag_to_follow=None, save=True, data = datas[2], ncdf=True, folder=folder, posSize_small=False, legend=False, cbar=[False, False, False], cbarax=None)
    _ = obs.observe(1500, fig=None, ax=None, specific_ag_to_follow=None, save=True, data = datas[2], ncdf=True, folder=folder, posSize_small=False, legend=False, cbar=[False, False, False], cbarax=None)
    _ = obs.observe(1700, fig=None, ax=None, specific_ag_to_follow=None, save=True, data = datas[2], ncdf=True, folder=folder, posSize_small=False, legend=False, cbar=[False, False, False], cbarax=None)
    _ = obs.observe(1800, fig=None, ax=None, specific_ag_to_follow=None, save=True, data = datas[2], ncdf=True, folder=folder, posSize_small=False, legend=False, cbar=[False, False, False], cbarax=None)


# %%


# %% [markdown]
# # T PREF PLOT

# %%
for u in range(2):
    print(u)
    fig =plt.figure(figsize=(16,9))
    rcParams = {'font.size':20}
    plt.rcParams.update(rcParams)
    ax = []
    ax.append(fig.add_subplot(111))
    ax[0].set_xlim(datas[0].time[0], datas[0].time[-1])
    for i in range(0):
        ax.append(ax[0].twinx())
    if u==1:
        color = "blue"
        Title = r"$\mathbf{pop}(t)$  [$10^3 \, {\rm People}$]"
        TitleText = "Population Size"
        ylimhigh = 20


    else:
        Title=r"$\mathbf{T}(t)$  [$10^6\, {\rm Trees}$]"
        TitleText="Trees"
        ylimhigh=12
        color="green"
    styles = ["-", "--", ":", "-."]
    labels = ["Linear (Standard)", "Delayed", "Early", "Logistic"]
    for n,datas in enumerate([stds, delayed, careful, logistic]):
        style = styles[n]
        if u==1:
            datavals = [(datas[k].pop_ct.sum(dim='triangles')*1e-3) for k in range(len(datas))]
        if u==0:
            datavals = [(datas[k].T_ct.sum(dim='triangles')*1e-6) for k in range(len(datas))]
            
        mean = np.mean(datavals, axis=0)
        std = np.std(datavals, axis=0)
        ind = 0; 
        #label = r"$\mathbf{T}(t)$  [$10^6\, {\rm Trees}$]"
        label=labels[n]
        #for k in range(5):
        #    ax[ind].plot(datas[0].time,  datavals[k], alpha = 0.2, color=colors[ind])
        ax[ind].fill_between(datas[0].time, mean-std,  mean +std, color=color, alpha=0.1)
        b = ax[ind].plot(datas[0].time, mean, linestyle = style, color=color, lw=3, label=label)#, color=colors[0])
        ax[ind].set_ylabel(Title)



    for i,a in enumerate(ax):
        #a.spines["right"].set_position(('axes',1.0))
        a.tick_params(axis="y", colors=color)
        a.spines["right"].set_position(("axes", 1.0+(i)*0.1))
        #a.set_ylabel(["R", "L", "E", "V"][i])
        a.yaxis.label.set_color(color)
        a.set_ylim(0,ylimhigh)
        a.yaxis.set_label_position("right")
        a.yaxis.tick_right()
        a.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax[0].set_xlabel(r"Year  [${\rm A.D.}$]")
    ax[0].set_title(TitleText, x=0.01, y=0.2,fontweight="bold", loc='left')
    ax[0].set_xlim(datas[0].time.sel(time=1550),datas[0].time.sel(time=1900))
    plt.legend()
    plt.savefig("TPrefAdaption_"+TitleText+".pdf", bbox_inches="tight")


# %%



# %%
fig =plt.figure(figsize=(16,9))
rcParams = {'font.size':20}
plt.rcParams.update(rcParams)
ax = []
ax.append(fig.add_subplot(111))
ax[0].set_xlim(datas[0].time[0], datas[0].time[-1])
for i in range(0):
    ax.append(ax[0].twinx())

color = "red"
#Title=r"$\mathbf{T}(t)$  [$10^6\, {\rm Trees}$]"
Title =r"Burnt Trees  [$\frac{10^3\, {\rm Trees}}{20\, {\rm yrs}}$]"

TitleText = "Burnt Trees"
ylimhigh = 12


styles = ["-", "--", ":", "-."]
labels = ["Linear (Standard)", "Delayed", "Early", "Logistic"]
for n,datas in enumerate([stds, delayed, careful, logistic]):
    style = styles[n]
    datavals = [(datas[k].T_ct.sum(dim='triangles')*1e-6) for k in range(len(folders))]
    mean = np.mean(datavals, axis=0)
    std = np.std(datavals, axis=0)
    ind = 0; 
    #label = r"$\mathbf{T}(t)$  [$10^6\, {\rm Trees}$]"
    label=labels[n]
  
    datavals=[]
    for j in range(0,len(datas)):
        fireSize = np.zeros(len(datas[0].time))
        t0=datas[0].isel(time=0).time.values
        fireSize[datas[j].firesTime- t0] += datas[j].firesSize
        BINwidth=20
        bins = np.arange(t0, datas[0].isel(time=-1).time+1, step=BINwidth )
        binned_fireSize = [1e-3 * np.sum( fireSize[( bins[k]-t0)  :  (bins[k+1]-t0 )]) for k in range(len(bins[:-1]))]
        #ax[3].bar(bins[:-1], binned_fireSize,color=colors[3], width=BINwidth)#, bins=np.arange(data.time[0],data.time[-1]+1, step=20), alpha=1, color=colors[3])
        #ax[3].set_ylabel("Fires [1000 Trees /"+str(BINwidth)+" yrs]")
        #plt.plot(bins[:-1], binned_fireSize, lw=1, color=colors[3])
        datavals.append(binned_fireSize)
    mean = np.mean(datavals, axis=0)
    std = np.std(datavals, axis=0)

    ax[ind].fill_between(bins[:-1], mean-std,  mean +std, color=color, alpha=0.1)
    c = ax[ind].plot(bins[:-1], mean, linestyle = style, color=color, lw=3, label=label)#, color=colors[0])

    ax[ind].set_ylabel(Title)
for i,a in enumerate(ax):
    #a.spines["right"].set_position(('axes',1.0))
    a.tick_params(axis="y", colors=color)
    a.spines["right"].set_position(("axes", 1.0+(i)*0.1))
    #a.set_ylabel(["R", "L", "E", "V"][i])
    a.yaxis.label.set_color(color)
    a.set_ylim(0,ylimhigh)
    a.yaxis.set_label_position("right")
    a.yaxis.tick_right()
    a.yaxis.set_major_locator(MaxNLocator(integer=True))

ax[0].set_xlabel(r"Year  [${\rm A.D.}$]")
ax[0].set_title(TitleText, x=0.01, y=0.2,fontweight="bold", loc='left')
#ax[0].set_xlim(datas[0].time.sel(time=1600),datas[0].time.sel(time=))
plt.legend(loc="upper left")

TotalBurntTrees=[]
for datas in [stds, delayed, careful, logistic]:
    TotalBurntTrees.append( np.mean([datas[k].firesSize.sum() for k in range(15)]))
Texts = ["%.1f" % (TotalBurntTrees[k]*1e-6)+r"$10^{-6}$" for k in range(4)]
print(Texts)
plt.savefig("TPrefAdaption_BurntTrees.pdf", bbox_inches="tight")


# %%
TotalBurntTrees=[]
for datas in [stds, delayed, careful, logistic]:
    TotalBurntTrees.append( np.mean([datas[k].firesSize.sum() for k in range(15)]))
Texts = ["%.1f" % (TotalBurntTrees[k]*1e-6)+r"$10^{-6}$" for k in range(4)]
print(Texts)


# %%


# %% [markdown]
# # Population Growth

# %%
len(stds)


# %%
k = 2
fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111)
netgrowths = [100*(stds[k].total_excessbirths- stds[k].total_excessdeaths)/stds[k].pop_ct.sum(dim="triangles") for k in range(15) ]

#ax.fill_between(stds[0].time, np.mean(netgrowths, axis=0)- np.std(netgrowths, axis=0), np.mean(netgrowths, axis=0)+ np.std(netgrowths, axis=0), alpha=0.2, color="blue")


N = 20
for a in netgrowths:
    ax.plot(stds[0].time, a, alpha = 0.1, color="blue")
ax.plot([], [], alpha = 0.1, color="blue", label="Net Growth Rate (Single Run)")

x = np.mean(netgrowths, axis=0)
ax.plot(stds[0].time, x, alpha = 1, color="blue", label="Net Growth Rate (Ensemble Mean)")
ax.plot(np.convolve(stds[0].time, np.ones((N,))/N, mode='valid'),np.convolve(x, np.ones((N,))/N, mode='valid'), lw=3, label='     -"-    Running Mean (N=20)', color="red")
ax.set_xlabel(r"Years  [${\rm A.D.}$]")
ax.set_ylabel(r"Net Growth Rate [$\%$] of the Population Size")
ax.set_ylim(-2, 2)
plt.legend(loc="lower left")
ax.set_title("Growth Rate (Standard Run)", x=0.95, y=0.9,fontweight="bold", loc='right')
ax2=ax.twinx()
ax2.set_yticks([-2, -1.5, -1, -0.5, 0])
ax2.set_ylim(-2,2)
ax.set_yticks([0, 0.5, 1.0,1.5, 2.0])
ax.axhline(0,0,1, color="black")
ax.set_xlim(800,1900)
plt.savefig(folder_std +    "NetGrowthRate.pdf", bbox_inches = "tight")

# %% [markdown]
# # SECONDARY STATS 1 NOT USED

# %%
fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111)

ax.fill_between(datas[0].time, datas[0].happyMeans-datas[0].happyStd,datas[0].happyMeans+datas[0].happyStd, color="blue", alpha=0.2)
ax.plot(datas[0].time, datas[0].happyMeans, lw=3, color="blue", label=r"Mean $h_{\rm i}(t)$")
ax.plot([],[], color="white", label = "Fraction of agents with ...")
ax.plot(datas[0].time, datas[0].farmingFills, lw=3, color="orange", label="...insufficient farming")
ax.plot(datas[0].time, datas[0].treeFills, lw=3, color='green', label="...insufficient tree harvest")
ax.set_xlim(datas[0].time.sel(time=1500), datas[0].time[-1])
ax.set_ylim(0,1)
ax.set_xlabel(r"Years  $[{\rm A.D.}]$")
plt.legend(loc="center left")

# %% [markdown]
# # SECONDARY STATS USED

# %%
fig = plt.figure(figsize=(16,9))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
k = 12
datas = stds
Nr_agents = datas[k].TreePrefAgents.count(dim = "index")

#ax1.fill_between(datas[k].time, datas[k].happyMeans-datas[k].happyStd,datas[k].happyMeans+datas[k].happyStd, color="blue", alpha=0.1)
a, = ax1.plot(datas[k].time, datas[k].happyMeans, lw=3, color="blue", label=r"Mean Happiness $\overline{h_{\rm i}}(t)$")
b00, = ax2.plot(datas[k].time, Nr_agents, "--", lw=3, color="black", label=r"Nr of total Agents")
b0, = ax2.plot([],[], color="white", label = "Nr of agents with ...")
b, = ax2.plot(datas[k].time, datas[k].farmingFills* Nr_agents, "--", lw=3, color="orange", label="...insufficient farming")
c, = ax2.plot(datas[k].time, datas[k].treeFills * Nr_agents, "--", lw=3, color='green', label="...insufficient tree harvest")
#d, = ax2.plot(datas[k].time, datas[k].Nr_Moves, lw=3, color='gray', label="Nr of agents moving")
Fprime = datas[k].Fraction_eroded
e, = ax1.plot(datas[k].time, Fprime/(1+Fprime), lw=3, color="brown", label="Fraction of eroded, \n (previously) well-suited cells")
f, = ax1.plot(datas[k].time, datas[k].GardenFraction, lw=3, color="cyan", label="Fraction of all farms on \npoorly suited cells")
ax1.set_xlim(datas[k].time.sel(time=1350), datas[k].time[-1])
ax1.set_ylim(0,1)
ax2.set_ylim(0,)
ax2.set_ylabel("Nr of Agents (dashed lines)")
ax1.set_ylabel(r"Fraction and Mean Happiness")
ax1.set_xlabel(r"Years  $[{\rm A.D.}]$")
plots = [a,b00,b0,b,c, e, f]
plt.legend(plots, [x.get_label() for x in plots], loc="upper left")
plt.savefig("STDsecondaryStats.pdf", bbox_inches="tight")


# %%


# %% [markdown]
# # TREE PREF

# %%
colors= np.array([
    (27,158,119),
    (217,95,2),
    (117,112,179),
    (231,41,138)
])/255
styles = ["-", "--", ":", "-."]
labels = ["Linear (Standard)", "Delayed", "Early", "Logistic"]


# %%
delayedTPref = [delayed[j].TreePrefAgents.mean(dim="index") for j in range(15)]
standardTPref = [stds[j].TreePrefAgents.mean(dim="index") for j in range(15)]
carefulTPref = [careful[j].TreePrefAgents.mean(dim="index") for j in range(15)]
logisticTPref = [logistic[j].TreePrefAgents.mean(dim="index") for j in range(15)]


# %%
fig=plt.figure(figsize=(16,9))
ax = fig.add_subplot(111)


m = np.mean(standardTPref, axis=0)
s = np.std(standardTPref, axis=0)
ax.fill_between(stds[0].time, m-s, m+s, color=colors[0], alpha=0.2)
ax.plot(stds[0].time, m, linestyle = styles[0], lw=3, label="Linear (Standard)", color=colors[0])
m = np.mean(delayedTPref, axis=0)
s = np.std(delayedTPref, axis=0)
ax.fill_between(stds[0].time, m-s, m+s, color=colors[1], alpha=0.2)
ax.plot(stds[0].time, m, linestyle = styles[1], lw=3, color=colors[1], label="Delayed")
m = np.mean(carefulTPref, axis=0)
s = np.std(carefulTPref, axis=0)
ax.fill_between(stds[0].time, m-s, m+s, color=colors[2], alpha=0.2)
ax.plot(stds[0].time, m, linestyle = styles[2], color=colors[2], lw=3, label="Early")
m = np.mean(logisticTPref, axis=0)
s = np.std(logisticTPref, axis=0)
ax.fill_between(stds[0].time, m-s, m+s, color=colors[3], alpha=0.2)
ax.plot(stds[0].time, m, linestyle = styles[3], lw=3, color=colors[3], label="Logistic")

ax.set_xlabel(r"Year  [${\rm A.D.}$]")
ax.set_ylabel(r"$\overline{T_{\rm Pref, \ i}}(t)$")
ax.set_title("Mean Tree Preference", x=0.01, y=0.2,fontweight="bold", loc='left')
#ax[0].set_xlim(datas[0].time.sel(time=1600),datas[0].time.sel(time=))
plt.legend(loc="upper right")
ax.set_xlim(1200, 1900)
ax.grid()
plt.savefig("TPrefAdaption_TPref.pdf", bbox_inches="tight")

# %% [markdown]
# # Happy TPREF

# %%
delayedhappy= [delayed[j].happyMeans for j in range(15)]
standardhappy= [stds[j].happyMeans for j in range(15)]
carefulhappy = [careful[j].happyMeans for j in range(15)]
logistichappy = [logistic[j].happyMeans for j in range(15)]


# %%
fig=plt.figure(figsize=(16,9))
ax = fig.add_subplot(111)
m = np.mean(standardhappy, axis=0)
s = np.std(standardhappy, axis=0)
ax.fill_between(stds[0].time, m-s, m+s, color=colors[0], alpha=0.2)
ax.plot(stds[0].time, m, linestyle = styles[0], lw=5, label="Linear (Standard)", color=colors[0])
m = np.mean(delayedhappy, axis=0)
s = np.std(delayedhappy, axis=0)
ax.fill_between(stds[0].time, m-s, m+s, color=colors[1], alpha=0.2)
ax.plot(stds[0].time, m, linestyle = styles[1], lw=5, color=colors[1], label="Delayed")
m = np.mean(carefulhappy, axis=0)
s = np.std(carefulhappy, axis=0)
ax.fill_between(stds[0].time, m-s, m+s, color=colors[2], alpha=0.2)
ax.plot(stds[0].time, m, linestyle = styles[2], color=colors[2], lw=5, label="Early")
m = np.mean(logistichappy, axis=0)
s = np.std(logistichappy, axis=0)
ax.fill_between(stds[0].time, m-s, m+s, color=colors[3], alpha=0.2)
ax.plot(stds[0].time, m, linestyle = styles[3], lw=5, color=colors[3], label="Logistic")
ax.set_xlabel(r"Year  [${\rm A.D.}$]")
ax.set_ylabel(r"$\overline{h_{\rm i}}(t)$")
ax.set_title("Mean Happiness", x=0.01, y=0.2,fontweight="bold", loc='left')
#ax[0].set_xlim(datas[0].time.sel(time=1600),datas[0].time.sel(time=))
plt.legend(loc="upper right")
ax.set_xlim(1600, 1900)
ax.grid()
plt.savefig("TPrefAdaption_Happy.pdf", bbox_inches="tight")

# %% [markdown]
# 

# %%


# %% [markdown]
# # ALPHA 

# %%
folder_deterministic = "/home/peter/EasterIslands/Runs_22May/alphaDeterministic/"
folders = [folder_deterministic+"FullModel_grid50_gH17e-3_noRegrowth_highFix_linear_NormPop_alphaDeterministic_seed"+str(seed)+"/" for seed in np.arange(1,16)]
deterministic = [xr.open_dataset(folder+"Statistics.ncdf") for folder in folders]
label_deterministic = r"Moving `Optimal Location'"


# %%
folder_resource = "/home/peter/EasterIslands/Runs_22May/alphaResource/"
folders = [folder_resource+"FullModel_grid50_gH17e-3_noRegrowth_highFix_linear_NormPop_alphaResource_seed"+str(seed)+"/" for seed in np.arange(1,16)]
resource = [xr.open_dataset(folder+"Statistics.ncdf") for folder in folders]
label_Resource = r"Moving `Only Resources'"


# %%
folder_hopping = "/home/peter/EasterIslands/Runs_22May/alphaHopping/"
folders = [folder_hopping+"FullModel_grid50_gH17e-3_noRegrowth_highFix_linear_NormPop_alphaHopping_seed"+str(seed)+"/" for seed in np.arange(1,16)]
hopping = [xr.open_dataset(folder+"Statistics.ncdf") for folder in folders]
label_hopping = r"Moving `Trial\&Error Hopping'"


# %%
datas, label, folder = [deterministic,label_deterministic,folder_deterministic]
for datas, label, folder in zip([stds, deterministic, hopping, resource], [label_Standard1, label_deterministic, label_hopping, label_Resource], [folder_std, folder_deterministic, folder_hopping, folder_resource]):
    fig =plt.figure(figsize=(16,9))
    colors=["blue", "green", "red"]#, "orange"]
    ax = []
    ax.append(fig.add_subplot(111))
    ax[0].set_xlim(stds[0].time[0], stds[0].time[-1])
    for i in range(2):
        ax.append(ax[0].twinx())
    ylimhigh=[0 for i in range(3)]
    #datas = careful
    plot_stats(ax, label)

    #plt.legend([a,c,b])
    plt.savefig(folder+"EnsembleStatistics.pdf", bbox_inches="tight")
    plt.close()
    print(folder)
    _ = obs.observe(1000, fig=None, ax=None, specific_ag_to_follow=None, save=True, data = datas[2], ncdf=True, folder=folder, posSize_small=False, legend=False, cbar=[False, False, False], cbarax=None)
    _ = obs.observe(1500, fig=None, ax=None, specific_ag_to_follow=None, save=True, data = datas[2], ncdf=True, folder=folder, posSize_small=False, legend=False, cbar=[False, False, False], cbarax=None)
    _ = obs.observe(1700, fig=None, ax=None, specific_ag_to_follow=None, save=True, data = datas[2], ncdf=True, folder=folder, posSize_small=False, legend=False, cbar=[False, False, False], cbarax=None)
    _ = obs.observe(1800, fig=None, ax=None, specific_ag_to_follow=None, save=True, data = datas[2], ncdf=True, folder=folder, posSize_small=False, legend=False, cbar=[False, False, False], cbarax=None)

# %% [markdown]
# # TREQ, R_T

# %%
folder_largeRad= "/home/peter/EasterIslands/Runs_22May/largeRad/"
folders = [folder_largeRad+"FullModel_grid50_gH17e-3_noRegrowth_highFix_linear_NormPop_alphaStd_seed"+str(seed)+"/" for seed in np.arange(1,16)]
largeRad = [xr.open_dataset(folder+"Statistics.ncdf") for folder in folders]
label_largeRad = r"Larger Resource Search Radii"


# %%
folder_smallRad= "/home/peter/EasterIslands/Runs_22May/smallRad/"
folders = [folder_smallRad+"FullModel_grid50_gH17e-3_noRegrowth_highFix_linear_NormPop_alphaStd_seed"+str(seed)+"/" for seed in np.arange(1,16)]
smallRad = [xr.open_dataset(folder+"Statistics.ncdf") for folder in folders]
label_smallRad = r"Smaller Resource Search Radii"


# %%
folder_LargeTReq= "/home/peter/EasterIslands/Runs_22May/LargeTReq/"
folders = [folder_LargeTReq+"FullModel_grid50_gH17e-3_noRegrowth_highFix_linear_NormPop_alphaStd_seed"+str(seed)+"/" for seed in np.arange(1,16)]
LargeTReq = [xr.open_dataset(folder+"Statistics.ncdf") for folder in folders]
label_LargeTReq = r"Large $T_{\rm Req, \ pP}$"


# %%
for datas, label, folder in zip([smallRad,largeRad, LargeTReq], [label_smallRad, label_largeRad,label_LargeTReq], [folder_smallRad, folder_largeRad, folder_LargeTReq]):
    fig =plt.figure(figsize=(16,9))
    colors=["blue", "green", "red"]#, "orange"]
    ax = []
    ax.append(fig.add_subplot(111))
    ax[0].set_xlim(stds[0].time[0], stds[0].time[-1])
    for i in range(2):
        ax.append(ax[0].twinx())
    ylimhigh=[0 for i in range(3)]
    #datas = careful
    plot_stats(ax, label)

    #plt.legend([a,c,b])
    plt.savefig(folder+"EnsembleStatistics.pdf", bbox_inches="tight")
    plt.close()
    print(folder)
    _ = obs.observe(1000, fig=None, ax=None, specific_ag_to_follow=None, save=True, data = datas[2], ncdf=True, folder=folder, posSize_small=False, legend=False, cbar=[False, False, False], cbarax=None)
    _ = obs.observe(1500, fig=None, ax=None, specific_ag_to_follow=None, save=True, data = datas[2], ncdf=True, folder=folder, posSize_small=False, legend=False, cbar=[False, False, False], cbarax=None)
    _ = obs.observe(1700, fig=None, ax=None, specific_ag_to_follow=None, save=True, data = datas[2], ncdf=True, folder=folder, posSize_small=False, legend=False, cbar=[False, False, False], cbarax=None)
    _ = obs.observe(1800, fig=None, ax=None, specific_ag_to_follow=None, save=True, data = datas[2], ncdf=True, folder=folder, posSize_small=False, legend=False, cbar=[False, False, False], cbarax=None)


# %%


