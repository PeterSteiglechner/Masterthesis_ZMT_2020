##########################################
#####    CONFIG FILE FOR FULL MODEL   ####
##########################################

import numpy as np
class EI_empty:
    def __init__(self):
        # CONSTANT
        self.N_els=0
        self.corners={'upper_left':[0,0],'bla':[0,0]}
        self.EI_midpoints=[]
        self.pixel_dim=[]
        self.points_EI_km=np.array([[0,0],[0,0]])
        self.distMatrix=np.zeros([self.N_els, self.N_els])
        self.all_triangles = []
        self.mask = []
        self.EI_triangles=[]
        self.carrying_cap=np.array([])
        self.EI_triangles_areas=[]
        self.EI_midpoints_elev=[]
        self.EI_midpoints_slope=[]
        self.NeighbourhoodRad = 0# = config.params['resource_search_radius']
        self.TreeNeighbours_of_triangles =[]
        self.AgricNeighbours_of_triangles =[]
        # CONSTANT EXTENSIONS
        self.water_penalties=[]
        self.map_penalty = np.array([])
        self.slopes_cond = np.array([])
        self.water_triangle_inds = []
        self.nr_highqualitysites = []
        self.nr_lowqualitysites = []
        self.agric_yield=np.array([])
        self.treeless_years = np.array([])
        self.max_tree_density = 0
        
        # WILL CHANGE OVER TIME
        self.agentOccupancy=np.array([])
        self.populationOccupancy = np.array([])
        self.tree_density=np.array([])
        self.agriculture = np.array([])
        self.AnakenaBeach_ind=-1

    def transform(self, pos):
        return pos
    def get_triangle_of_point(self, pos, triObject):
        ind_triangle, m, _corner = [0,0,[]]
        return ind_triangle, m, _corner
    def plot_TreeMap_and_hist(self , ax=None, fig=None, name_append = "", hist=False, save=False, histsavename="", folder="Map/"):
        return fig, ax
    def plot_agricultureSites(self, ax=None, fig=None, save=True, CrossesOrFacecolor="Crosses", folder=""):
        return fig, ax

### HOUSEHOLD PARAMS

MaxSettlementSlope=0
#MaxSettlementElev=0
#MinSettlementElev=0
Penalty50_SettlementElev = 0


max_pop_per_household_mean = 0
max_pop_per_household_std = 0
LowerLimit_PopInHousehold = 0
childrenPop = 0


MaxPopulationDensity=0
MaxAgricPenalty = 0

tree_need_per_capita = 0
MinTreeNeed=0. #Fraction
TreePref_kappa = 0

BestTreeNr_forNewSpot = 0

#Nr_AgricStages = 1
#dStage = ((1-MinTreeNeed)/Nr_AgricStages)

agricYield_need_per_Capita = 0

maxNeededAgric=0

#treePref_decrease_per_year = 0.0
#treePref_change_per_BadYear = 0.0

#HowManyDieInPopulationShock = 0
FractionDeathsInPopShock=0.0

alpha_w=0
alpha_t=0
alpha_p=0
alpha_a=1
alpha_m= 0

PenalToProb_Prefactor=0

NrOfEquivalentBestSitesToMove = 0

# Each agent can (in the future) have these parameters different
params={'tree_search_radius': 0, 
        'agriculture_radius':0,
        'moving_radius': 0,
        'reproduction_rate': 0.,
        }
PopulationMovingRestriction=0

MaxFisherAgents = 0


###  INIT
N_agents = 0
N_trees = 0
firstSettlers_moving_raidus = 0
firstSettlers_init_pop=0
init_TreePreference = 1.
index_count=0  # Index of agent (index is unique regardless of dead households)


### RUNS ###
folder = ""
seed = 100
updatewithreplacement = True
StartTime = 0
EndTime=0
N_timesteps = 0
move=0
#t = 800

###  MAP  ###
TreeDensityConditionParams = {'minElev':0, 'maxElev':0,'maxSlope':0}
tree_pop_percentage = 0.0
tree_pop_timespan  = 0
UpperLandSoilQuality = 0
ErodedSoilYield = 0 # Max
#YearsBeforeErosionDegradation=0
#AgriConds={'minElev':0,'maxElev_highQu':0,
#    'maxSlope_highQu':0,'maxElev_lowQu':0,'maxSlope_lowQu':0,
#    'MaxWaterPenalty':0,}
m2_to_acre = 0.000247105381  # = acre per m^2
km2_to_acre = m2_to_acre*1e6 # = acre per km^2 --> areas*km2_to_acre 
gridpoints_y= 0
AngleThreshold = 0
MapPenalty_Kappa = 0

tree_regrowth_rate = 0

drought_RanoRaraku_1=[]

EI_triObject = 0
tree_growth_poss=np.array([],dtype=int)
EI=EI_empty()

analysisOn= True
AG0_mv_inds=np.array([])
AG0_total_penalties=np.array([])

AG0_water_penalties =np.array([])
AG0_tree_penalty =np.array([])
AG0_pop_density_penalty = np.array([])
AG0_agriculture_penalty = np.array([])
AG0_nosettlement_zones = np.array([])
AG0_MovingProb=np.array([])
AG0_map_penalty = np.array([])




#######################
###  RESULT ARRAYS  ###
#######################
agents=[]
nr_agents, nr_trees, nr_deaths, nr_pop = [[], [], [], []] #[nr_agents, nr_trees, nr_deaths, population]
deaths = 0
fire=[]
agents_stages=[]
agents_pos_size = []
nr_happyAgents=[]
nr_agricultureSites=[]
Array_tree_density = np.array([]).astype(np.int32)
Array_agriculture = np.array([]).astype(np.int32)
Array_populationOccupancy = np.array([]).astype(np.int32)
Penalty_std = np.array([])
Penalty_mean = np.array([])
FisherAgents = 0
NrFisherAgents = np.array([])
Fraction_eroded=np.array([])
GardenFraction = np.array([])
moves = []
TreeRegrowth = []
TreePopUp=[]
timeSwitchedMovingRad = np.nan