
import numpy as np
from time import time
from copy import copy
import config


# UPDATE
def update_time_step():
    ''' update one time step, i.e. perform update_single_agent() N_agents times'''
    #global nr_agents, nr_trees, nr_deaths
    #global EI

    start_step = time()
    N_agents = len(config.agents)
    if not N_agents==0:
        order = np.random.choice(np.arange(0,N_agents), N_agents, replace=False).astype(int)
        agents_order = [config.agents[o] for o in order]
        for n in range(N_agents): 
            # Select random agent
            if config.updatewithreplacement == True:
                # Problem: what if agents die!!??
                #ind_agent = order[n]
                selected_agent=agents_order[n]
            else:
                ind_agent = np.random.randint(0,N_agents)
                selected_agent = config.agents[ind_agent]

            # Update this agent
            update_single_agent(selected_agent)
    
    
    
    config.nr_agents.append(len(config.agents))
    currentNrtrees = (np.sum(config.EI.tree_density) if config.map_type=="EI" else len(config.trees))
    config.nr_trees.append(currentNrtrees)
    if config.agent_type == "household":
        config.nr_pop.append(np.sum([ag.pop for ag in config.agents]))
    else:
        config.nr_pop.append(len(config.agents))
    config.nr_deaths.append(config.deaths)
    config.deaths = 0
    print("Run ",len(config.nr_agents),"  Stats: ",config.nr_agents[-1], config.nr_trees[-1], 
    config.nr_deaths[-1], config.nr_pop[-1], "\t Time tot: ", '%.2f'  % (time()-start_step))

    
    return 

def update_single_agent(ag):
    ''' for agent `ag': perform update step ''' 
    #global agents, deaths, EI
    
    ####################
    # search for food  #
    ####################
    if config.map_type=="EI":
        trees_nearby, nb_trees = ag.search_resource_treedensity()  # returns a list of nearby trees next to Agent ag
    elif config.map_type=="unitsquare":
        trees_nearby, nb_trees= ag.search_resource_treeAgent()
    else:
        config.wrongMapType()


    ################################
    ## Decision Eat, Move, or Die ##
    ################################
    if config.agent_type=="individual":
        unhappyCondition = (trees_nearby==0)
    elif config.agent_type =="household":
        unhappyCondition= (trees_nearby<ag.tree_need+1)
    else:
        config.wrongAgentType()

    if unhappyCondition: # Not enough Food Found
        if ag.hungry:
            # die
            config.deaths+=1
            if config.map_type=="EI": 
                config.EI.agentOccupancy[ag.triangle_ind] -=1
            config.agents.remove(ag)
            return
        else:
            # Phew... not dead yet, move and try next spot. But now hungry.
            ag.move_randomHop()
            ag.hungry = True
        
    else: # enough food found
        
        ### 1. Eat some tree
        if config.agent_type=="individual":
            eat(nb_trees, nr=1)
        else:
            p = ag.tree_need
            decimal = p-int(p)
            if np.random.random()<decimal:
                eat(nb_trees, nr = int(p)+1)
            else:
                eat(nb_trees, nr = int(p))     
        
        ag.hungry = False

        ### 2. Perhaps Reproduce
        Reproduction(ag)
       
    return 
        

def eat(nb_trees, nr = 1):
    # select random tree or tree index nearby
    for n in range(nr):
        cut_tree_ind = np.random.choice(nb_trees, 1)
        # Eat that tree
        if config.map_type=="EI":
            config.EI.tree_density[cut_tree_ind]-=1
            if config.EI.tree_density[cut_tree_ind]==0:
                nb_trees.remove(cut_tree_ind)
        else:
            config.trees.remove(cut_tree_ind)
            nb_trees.remove(cut_tree_ind)
    return 

def Reproduction(ag):
    if config.agent_type=="individual":
        if np.random.random() < ag.fertility:
            child = copy(ag)
            child.move_randomHop()
            config.agents.append(child)
    else:
        ag.pop += np.round(ag.reprod_rate*ag.pop)
        if ag.pop > ag.max_pop_per_household:        #SPLIT
            child = copy(ag)
            child.pop = np.floor(ag.pop/2)
            ag.pop = np.ceil(ag.pop/2)
            child.calc_tree_need()
            child.move_randomHop()
            config.agents.append(child)      
        ag.calc_tree_need()

    return