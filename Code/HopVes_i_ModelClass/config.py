
import numpy as np
agents=[]
class EI_empty:
    def __init__(self):
        self.N_els=0
        self.tree_density=[]
        self.corners={'upper_left':[0,0],'bla':[0,0]}
        self.EI_midpoints=[]
        self.agentOccupancy=[]
        self.pixel_dim=[]
        self.points_EI_m=np.array([[0,0],[0,0]])
        self.EI_triangles=[]
        self.carrying_cap=[]
    def transform(self, pos):
        return pos
    def get_triangle_of_point(self, pos):
        ind_triangle, m, _corner = [0,0,[]]
        return ind_triangle, m, _corner

EI=EI_empty()
params_individual={}
params_hh={}
params={}
folder = ""
trees=[]


updatewithreplacement = True

map_type = ""
agent_type= ""
init_option = ""

gridpoints_y= 0

N_agents = 0
index_count=0
N_trees = 0
tree_regrowth_rate = 0

N_timesteps = 0

nr_agents, nr_trees, nr_deaths, nr_pop = [[], [], [], []] #[nr_agents, nr_trees, nr_deaths, population]
deaths = 0


TreeDensityConditionParams = {'minElev':0, 'maxElev':0,'maxSlope':0}

def wrongMapType():
    print("ERROR: Pick a map_type of 'EI' or 'unitsquare'")
    quit()
def wrongConfigOption():
    print("ERROR: Pick a config_option of 'equally' or 'anakena' (if map_type =='EI') or 'corner' (if map_type=='unitsquare')")
    quit()
def wrongAgentType():
    print("ERROR: Pick a Agent_type of 'individual' or 'household'")
    quit()
