from pylab import *
import copy as cp
import random

T = 50             # simulation time limit
tree_init = 1000 # initial number of trees
house_init = 3      # initial number of households
in_pop = 20         # initial population per household

m = 0.35             # magnitude of movement of households
rr = 0.06            # population reproduction rate
tree_regrowth = 0.01 # tree regrowth rate [Not used]
cd = 0.06            # radius for tree harvesting

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
        proj_tree = project_on_triangle(np.array([tr.x,tr.y]))
        tr.x = proj_tree[0]
        tr.y = proj_tree[1]
        trees.append(tr)            # append (i.e. add) the ith agent into the array 'tree'

    for j in range(house_init):
        hm = household()
        hm.population = in_pop
        hm.penality_year = False
        hm.x = random.uniform(0, 0.05)
        hm.y = random.uniform(0, 0.05)
        proj_hm = project_on_triangle(np.array([hm.x,hm.y]))
        hm.x = proj_hm[0]
        hm.y = proj_hm[1]
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
    tot_pop = sum(np.array(population))
    #for i in range(len(population)):
    #    tot_pop = tot_pop + population[i]

    if len(households) > 0:
        x = [hm.x for hm in households]
        y = [hm.y for hm in households]
        plot(x, y, 'bo') #markersize = hm.population)

    treedata.append(len(trees))
    housedata.append(tot_pop)
    print(len(trees), tot_pop)
    axis('image')
    axis([0, 1, 0, 1])
    title('Z' + str(t))
    savefig('Z' + str(t)+ '.png')


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
    savefig(str(t) + '.png')

##### TRIANGLE ADJUSTMENT ## PS #######   
triangle_points = (np.array([0,0]), np.array([0,1]), np.array([1,1])) 
import scipy.optimize
def q(a,b):
	# this gives the point on the line at which I mirror, given points a,b
	v = (b-a)/np.sqrt(np.dot((b-a),(b-a)))
	return lambda s: a+s*v
def scalarprod(s,a,b,p):
	Q = q(a,b)(s)
	# Note I minimimize this function below. Why 2 terms? Because if only first term, there's a minimum if Q==a. Which will be chosen
	return np.dot(Q-a,Q-p) + np.dot(Q-b,Q-p) 

def project_on_triangle(p):
	global triangle_points
	a,b,c = triangle_points
	#solve via barycentric coordinates
	def barycentric(x,a,b,c,p):
		return p - (a + (b - a) * x[0] + ( c- a) * x[1])
	x0 = np.array([0.01, 0.2])
	x = scipy.optimize.root(barycentric, x0, args=(a,b,c,p,))
	s = x.x[0]
	t = x.x[1]
	#print("Barycentric coordinates: ", '%.2f' % s, ", ", '%.2f' % t)
	#print("    Point retrieved is: ", a + (b - a) * s + ( c- a) * t , " with p-bary = ", barycentric([s,t],a,b,c,p))
	if (s>=0) and (s<=1) and (t>=0) and (t<=1) and (s+t<=1):
		#print("In triangle")
		return p
	else:
		#print("Projected")
		# Find mirror points 
		mirrorpts=[]
		for n,(p1,p2) in enumerate([(a,b), (a,c), (b,c)]):
			vector = (p2-p1)/np.sqrt(np.dot((p2-p1), (p2-p1)))
			#print("Vector: ", vector)
			s0=0.001
			s = scipy.optimize.root(scalarprod, s0, args=(p1,p2,p))
			Q = q(p1,p2)(s.x)
			#print("Optimal s: ", s.x, " giving Q =", Q)
			#p_mirror = p+2*(Q-p)		
			mirrorpts.append(Q)
		#print(mirrorpts)
		distances = np.array([np.dot(qc- p, qc-p) for qc in mirrorpts])
		minimum = np.min(distances)
		#print(distances)
		projected_point = mirrorpts[np.where(distances==minimum)[0][0]]
		#print("Proje Point", projected_point)
		return projected_point
    #print("ERROR")
	return

def update():
    global trees, households

    cut_trees = []

    if households == []:
        return

    hm = households[randint(len(households))]

    # nearest trees
    neighbour_trees = [nt for nt in trees if (hm.x - nt.x) ** 2 + (hm.y - nt.y) ** 2 < cd**2]

    if len(neighbour_trees) == 0:
            if hm.penality_year == True:
                households.remove(hm)
            else:
                hm.x += uniform(-m, m)
                hm.y += uniform(-m, m)
                
                #### TRIANGLE ADJUSTMENT ## PS ######
                #hm.x = 1 if hm.x > 1 else 0 if hm.x < 0 else hm.x ### Doesn't this push households to the edges? 
                #hm.y = 1 if hm.y > 1 else 0 if hm.y < 0 else hm.y
                projected_pt = project_on_triangle(np.array([hm.x, hm.y]))
                hm.x    =projected_pt[0]
                hm.y=projected_pt[1]
                
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
        if len(households)!=0:
            tt += 1.0 / len(households)
        else:
	        #tt = 1.0
            break;
        update()

t = 0
initialize()
observe()

for t in range(1, T):
    update_one_unit_time()
    observe()
