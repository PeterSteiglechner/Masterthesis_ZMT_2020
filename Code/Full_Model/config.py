import numpy as np

N_agents_arrival = 0
Pop_arrival = 0
r_M_arrival = 0
t_arrival = 0
F_pi_c_tarrival = np.array([])


# MAP
gridpoints_y = 0
AngleThreshold = 0

N_trees_arrival = 0

TreeDensityConditionParams =   {
    'minElev':0, 
    'maxSlopeHighD': 0, 
    "ElevHighD":0, 
    'maxElev': 0,
    'maxSlope':0,
    "factorBetweenHighandLowTreeDensity":1
}

F_PI_well = 0
F_PI_poor = 0
F_PI_eroded = 0

tree_regrowth_rate = 0
tree_pop_percentage = 0
barrenYears_beforePopUp = 0

droughts_RanoRaraku=[[], []]

well_suited_cells_tarrival = np.array([], dtype = int)

lakeColor = (3/255,169/255,244/255 ,1 )
m2_to_acre = 0.000247105381  # = acre per m^2
km2_to_acre = m2_to_acre*1e6 # = acre per km^2 --> areas*km2_to_acre 



# Section Harvest Environment interaction

r_T = 0
r_F = 0
T_Req_pP = 0
T_Pref_max = 0
T_Pref_min = 0
xi_T_Pref = 0
T_Pref_fisher_min = 0
MaxFisherAgents = 0
F_Req_pP=0

#FishingTabooYear = 0


# Population dynamics

g_H1 = 0
H_equ = 0
SplitPop = 0
pop_agent_max_mean = 0
pop_agent_max_std=0
Split_k=0
Split_theta = 0
pop_agent_min = 0
g_shape = 0
g_scale = 0.1
g_scale_param = 0
g_of_H = lambda H: 0


# Moving 
r_M_init = 0
r_M_later = 0
pop_restrictedMoving = 0
k_X = lambda x01,x99: 1/(0.5*(x99-x01)) * np.log(0.99/0.01)
P_x = lambda x, k_x, x50: 1/(1+np.exp(-k_x*(x-x50)))  # x50 = (x99+x01)/2


# categories 
w01=0; w99=0
el01=0; el99=0
sl01=0; sl99=0
tr01=0; tr99=0 
f01=0; f99=0
pd01=0; pd99=0
trcondition = tr99
fcondition = f99

alpha_W = 0
alpha_T=0
alpha_F = 0
alpha_D = 0
alpha_G=0
gamma = 0


# RUN SETTINGS
t_end = 0
updatewithreplacement = False
seed = 0

##### RUNING VARIABLES / STORAGE #####

folder = ""
index_count = 0
agents = []

# STORAGE
Array_N_agents = np.array([]).astype(np.int32)
Array_pop_ct= np.array([]).astype(np.int)
Array_T_ct= np.array([]).astype(np.int32)
Array_F_ct = np.array([]).astype(np.int32)
agentStats_t = []

FisherAgents = 0
Pop = np.array([])
RegrownTrees = np.array([])
TreesPoppedUp = np.array([])
excess_deaths = 0
nr_excessdeaths =  np.array([])
excess_births = 0
nr_excessbirths =  np.array([])
fire=[]
happyMeans=np.array([])
happyStd=np.array([])
treeFills = np.array([])
farmingFills = np.array([])
Penalty_std = np.array([])
Penalty_mean = np.array([])
NrFisherAgents = np.array([])
Fraction_eroded=np.array([])
GardenFraction = np.array([])
moves = []
move=0
timeSwitchedMovingRad = np.nan


class EI_Empty:
    def __init__(self):
        self.N_c = 0
        self.T_c= np.array([])
        self.A_c= np.array([])
        self.A_F_c= np.array([])
        self.P_W= np.array([])
        self.P_G= np.array([])
        self.points_EI_km= np.array([])
        self.carrying_cap= np.array([])
        self.F_pi_c= np.array([])
        self.pop_c = np.array([])
        self.C_T_c= np.array([])
        self.C_F_c= np.array([])
        self.C_Mlater_c= np.array([])
        self.A_acres_c = np.array([])
        self.agNr_c = np.array([])
        self.vec_c = np.array([])
        self.points_EI_km=np.array([])
        self.EI_triangles = np.array([])
        self.AnakenaBeach_ind =0
        self.C_F_cAnakena=np.array([])
        self.well_suited_cells = np.array([], dtype=int)
        self.poorly_suited_cells = np.array([], dtype=int)
        self.eroded_well_suited_cells = np.array([], dtype=int)
        # for plotting
        self.corners={'upper left':[0,0]}
        self.water_triangle_inds_RarakuDrought = []
        self.water_triangle_inds_NoDrought = []
        self.r_T = 0
        self.r_F = 0
    def check_drought(self):
        return 
    def init_trees(self):
        return
    def get_c_of_point(self,point, triObject):
        ind = 0; vec_t_ind=[0,0]
        return ind, vec_t_ind 
    def get_P_G(self):
        return 
EI = EI_Empty()
EI_triObject = None


def logistic(x,k,mu):
    return 1/(1+np.exp(-k*(x-mu)))

f_T_Pref = lambda x: 0