#from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import copy as cp
import random
import scipy.optimize

path = "pics_v2_2/"
T = 150            # simulation time limit
tree_init = int(1e4) # initial number of trees
house_init = 3      # initial number of households
in_pop = 20         # initial population per household

m = 0.35             # magnitude of movement of households
rr = 0.06            # population reproduction rate
tree_regrowth = 0.01 # tree regrowth rate [Not used]
cd = 0.06            # radius for tree harvesting


triangle_points = (np.array([0.1,0.1]), np.array([0.05,0.9]), np.array([0.95,0.75])) 


def create_triangle_p(point):
    global triangle_points
    a,b,c = triangle_points
	#solve via barycentric coordinates
    def barycentric(x,a,b,c,p):
        return p - (a + (b - a) * x[0] + ( c- a) * x[1])
    x0 = np.array([0.01, 0.2])
    x = scipy.optimize.root(barycentric, x0, args=(a,b,c,point,))
    s = x.x[0]
    t = x.x[1]
    #print("Barycentric coordinates: ", '%.2f' % s, ", ", '%.2f' % t)
    #print("    Point retrieved is: ", a + (b - a) * s + ( c- a) * t , " with p-bary = ", barycentric([s,t],a,b,c,point))
    if (s>=0) and (s<=1) and (t>=0) and (t<=1) and (s+t<=1):
        #print("In triangle")
        prob = 1.
        return prob
    else:
        prob=0
        return prob

def plot_triangle():
    grid = np.linspace(0,1,num=51)
    global triangle_points
    a,b,c = triangle_points
    print(a,b,c)
    X,Y = np.meshgrid(grid, grid)
    Z = np.empty_like(X)
    for n in range(len(grid)):
        for m in range(len(grid)):
            #print(" POINT: ", np.array([X[n,m], Y[n,m]]))
            Z[n,m] = create_triangle_p(np.array([X[n,m], Y[n,m]]))
    plt.pcolormesh(X,Y,Z)
    plt.colorbar()
    plt.scatter(np.array([a[0], b[0], c[0]]), np.array([a[1], b[1], c[1]]),  marker = 'x')
    plt.show()
    plt.close()
plot_triangle()

class tree:
    pass
class household:
    pass

def initialize():
    global trees, households, treedata, housedata, triangle_points
    a,b,c = triangle_points
    trees = []
    households = []
    treedata = []
    housedata = []

    for i in range(tree_init):      #distribute the trees
        tr = tree( )
        tr.id = i
        #tr.x = random.random()      # randomly generates a number between 0.0 and 1.0 for x position
        #tr.y = random.random()      # randomly generates a number between 0.0 and 1.0 for y position

        while (1):
            point = np.array([np.random.uniform(), np.random.uniform()])
            if create_triangle_p(point) ==1:
                tr.x=point[0]
                tr.y=point[1]
                break

        trees.append(tr)            # append (i.e. add) the ith agent into the array 'tree'
    print("Init Trees: DONE")
    for j in range(house_init):
        hm = household()
        hm.population = in_pop
        hm.penality_year = False

        
        while (1):
            #init_x = np.random.uniform(0.0, 0.05)
            #init_y = np.random.uniform(0.1, 0.15)
            init_border = np.random.uniform(0.2, 0.3)
            #point = a+np.array([init_x, init_y])
            point = a+(b-a)*init_border
            if create_triangle_p(point) ==1:
                hm.x=point[0]
                hm.y=point[1]
                break
        households.append(hm)       # append (i.e. add) the ith agent into the array ’agents’
    print("Init Houses: DONE")

def observe():
    global trees, households, treedata, housedata
    plt.subplot(1, 1, 1)
    plt.cla()

    if len(trees) > 0:
        x = [tr.x for tr in trees]
        y = [tr.y for tr in trees]
        plt.plot(x, y, 'g.')

    population = [hm.population for hm in households]
    tot_pop = sum(np.array(population))
    #for i in range(len(population)):
    #    tot_pop = tot_pop + population[i]

    if len(households) > 0:
        x = [hm.x for hm in households]
        y = [hm.y for hm in households]
        plt.plot(x, y, 'bo') #markersize = hm.population)

    treedata.append(len(trees))
    housedata.append(tot_pop)
    print(len(trees), tot_pop)
    plt.axis('image')
    plt.axis([0, 1, 0, 1])
    plt.title('Z' + str(t))
    plt.savefig(path+'Z' + str(t)+ '.png')
    plt.close()


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

    plt.title(str(t))
    plt.savefig(path+str(t) + '.png')
    plt.close()

def update():
    global trees, households

    cut_trees = []

    if households == []:
        return

    hm = households[np.random.randint(len(households))]

    # nearest trees
    neighbour_trees = [nt for nt in trees if (hm.x - nt.x) ** 2 + (hm.y - nt.y) ** 2 < cd**2]

    if len(neighbour_trees) == 0:
            if hm.penality_year == True:
                households.remove(hm)
            else:
                while (1):
                    step = np.array([np.random.uniform(-m,m), np.random.uniform(-m,m)])
                    point = step+ np.array([hm.x, hm.y])
                    if create_triangle_p(point) ==1:
                        hm.x=point[0]
                        hm.y=point[1]
                        break
                hm.penality_year = True
            return

    else:  # if there are trees nearby
        hm.penality_year = False
        hm.population = round(hm.population + rr * hm.population)
        if hm.population > 100:
            hm.population = round(hm.population / 2)
            households.append(cp.copy(hm))
            while (1):
                step = np.array([np.random.uniform(-m,m), np.random.uniform(-m,m)])
                point = step+ np.array([hm.x, hm.y])
                if create_triangle_p(point) ==1:
                    hm.x=point[0]
                    hm.y=point[1]
                    break
            #hm.x = 1 if hm.x > 1 else 0 if hm.x < 0 else hm.x
            #hm.y = 1 if hm.y > 1 else 0 if hm.y < 0 else hm.y

        for i in range(hm.population):
            if len(neighbour_trees) > 0:
                x = np.random.randint(len(neighbour_trees))
                cut_trees.append(neighbour_trees[x])
                neighbour_trees.remove(neighbour_trees[x])

        trees = [elem for elem in trees if elem not in cut_trees]

def update_one_unit_time():
    global households
    tt = 0.0
    while tt < 1.0:
        if len(households)!=0:
            tt += 1.0 / len(households)
        else:
	        #tt = 1.0
            return False
            break
        update()

t = 0
initialize()
observe()

for t in range(1, T):
    running = update_one_unit_time()
    observe()
    if running==False: break
