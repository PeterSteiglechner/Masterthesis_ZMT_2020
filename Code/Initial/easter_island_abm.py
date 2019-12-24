from pylab import *
import copy as cp
import random

path = "pics_orig"

T = 600             # simulation time limit
tree_init = 1000000 # initial number of trees
house_init = 3      # initial number of households
in_pop = 20         # initial population per household

m = 0.35             # magnitude of movement of households
rr = 0.06            # population reproduction rate
tree_regrowth = 0.01 # tree regrowth rate [Not used]
cd = 0.06            # radius for tree harvesting
cdsq = cd ** 2

class tree:
    pass
class household:
    pass

def initialize():
    global trees, households, treedata, housedata
    trees = []
    households = []
    treedata = []
    housedata = []

    for i in range(tree_init):      #distribute the trees
        tr = tree( )
        tr.id = i
        tr.x = random.random()      # randomly generates a number between 0.0 and 1.0 for x position
        tr.y = random.random()      # randomly generates a number between 0.0 and 1.0 for y position
        trees.append(tr)            # append (i.e. add) the ith agent into the array 'tree'

    for j in range(house_init):
        hm = household()
        hm.population = in_pop
        hm.penality_year = False
        hm.x = random.uniform(0, 0.05)
        hm.y = random.uniform(0, 0.05)
        households.append(hm)       # append (i.e. add) the ith agent into the array ’agents’

def observe():
    global trees, households, treedata, housedata
    subplot(1, 1, 1)
    cla()

    if len(trees) > 0:
        x = [tr.x for tr in trees]
        y = [tr.y for tr in trees]
        plot(x, y, 'g.')

    population = [hm.population for hm in households]
    tot_pop = 0
    for i in range(len(population)):
        tot_pop = tot_pop + population[i]

    if len(households) > 0:
        x = [hm.x for hm in households]
        y = [hm.y for hm in households]
        plot(x, y, 'bo')

    treedata.append(len(trees))
    housedata.append(tot_pop)
    print(len(trees), tot_pop)
    axis('image')
    axis([0, 1, 0, 1])
    title('Z' + str(t))
    savefig(path+'Z' + str(t)+ '.png')


    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(treedata, label = 'Trees', color = 'g')
    ax2.plot(housedata, label = 'Population', color = 'b')

    ax1.set_xlabel('Years')
    ax1.set_xlim([0, T])
    ax1.set_ylabel('Trees', color = 'g')
    ax1.set_ylim([0, tree_init])
    ax2.set_ylabel('Population', color = 'b')
    ax2.set_ylim([0, 40000])

    title(str(t))
    savefig(path+str(t) + '.png')

def update():
    global trees, households

    cut_trees = []

    if households == []:
        return

    hm = households[randint(len(households))]

    # nearest trees
    neighbour_trees = [nt for nt in trees if (hm.x - nt.x) ** 2 + (hm.y - nt.y) ** 2 < cdsq]

    if len(neighbour_trees) == 0:
            if hm.penality_year == True:
                households.remove(hm)
            else:
                hm.x += uniform(-m, m)
                hm.y += uniform(-m, m)
                hm.x = 1 if hm.x > 1 else 0 if hm.x < 0 else hm.x
                hm.y = 1 if hm.y > 1 else 0 if hm.y < 0 else hm.y
                hm.penality_year = True
            return

    else:  # if there are trees nearby
        hm.penality_year = False
        hm.population = round(hm.population + rr * hm.population)
        if hm.population > 100:
            hm.population = round(hm.population / 2)
            households.append(cp.copy(hm))
            hm.x += uniform(-m, m)
            hm.y += uniform(-m, m)
            hm.x = 1 if hm.x > 1 else 0 if hm.x < 0 else hm.x
            hm.y = 1 if hm.y > 1 else 0 if hm.y < 0 else hm.y

        for i in range(hm.population):
            if len(neighbour_trees) > 0:
                x = randint(len(neighbour_trees))
                cut_trees.append(neighbour_trees[x])
                neighbour_trees.remove(neighbour_trees[x])

        trees = [elem for elem in trees if elem not in cut_trees]

def update_one_unit_time():
    global households
    tt = 0.0
    while tt < 1.0:
        tt += 1.0 / len(households)
        update()

t = 0
initialize()
observe()

for t in range(1, T):
    update_one_unit_time()
    observe()
