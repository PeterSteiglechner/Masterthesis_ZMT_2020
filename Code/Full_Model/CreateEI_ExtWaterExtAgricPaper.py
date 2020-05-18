#!/usr/bin/env python3


#Peter Steiglechner, 6.3.2020
## edited to fit Model Class on 9.3.2020
## edited to include a water penalty
## edited to be able to pickle.dump a object.
## edited to give nr of agriculture sites 25.03.2020
## edited to get agriculture not from elevation/slope but from Paper Puleston2017  02.04.2020
## edited for distance matrix 28.04.2020
## edited add fish in agriculture 29.04.2020
## adjusted Names and final changes 16.05.2020

#
#This script contains a class Map
#It sets up a Map which performs the triangulation, 
#    stores triangles (with midpoints, elevation, slope, ...) 
#    and variables (tree density, carrying_cap, agriculture_sites, water penalty)
#
#It uses two pictures of pixel size 900x600 of elevation and slope of Easter Island 
#These were created through Google Earth Enginge API
#
#USAGE:
#config.TreeDensityConditionParams = {'minElev':10, 'maxElev':430,'maxSlope':7}
#config.N_init_trees = 12e6
#config.gridpoints_y=100
#config.AgriConds={'minElev':20,'maxElev_highQu':250,'maxSlope_highQu':3.5,'maxElev_lowQu':380,'maxSlope_lowQu':6,'MaxWaterPenalty':300,}
#config.params= {"resource_search_radius":500}
#config.AngleThreshold = 0
#
#config.EI= Map(config.gridpoints_y, N_init_trees=config.N_init_trees, angleThreshold=config.AngleThreshold)
#triObject = mpl.tri.Triangulation(config.EI.points_EI_km[:,0], config.EI.points_EI_km[:,1], triangles = config.EI.all_triangles, mask=config.EI.mask)
#
#print(config.EI.get_triangle_of_point([1e4, 1e4], triObject))
#config.EI.plot_agriculture(which="high", save=True)
#config.EI.plot_agriculture(which="low", save=True)
#config.EI.plot_agriculture(which="both", save=True)
#config.EI.plot_TreeMap_and_hist(name_append="T12e6", hist=True, save=True)
#config.EI.plot_water_penalty()
#
#with open("EI_grid"+str(config.gridpoints_y), "wb") as store_file:
#    pickle.dump(config.EI,  store_file)
#
#with open("EI_grid"+str(config.gridpoints_y), "rb") as store_file:
#    EI_l = pickle.load(store_file)

import config
# config.py is the file that stores global variables such as EI, map_type (="EI"), 

import matplotlib as mpl 
import numpy as np
import matplotlib.pyplot as plt
import copy
import xarray as xr
import pandas as pd
#font = {'font.size':20}
#plt.rcParams.update(font)
import sys, os
import pickle

import scipy.spatial.distance


from mpl_toolkits.axes_grid1 import make_axes_locatable




def blockPrint():
    ''' Disable Printing '''
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    ''' Enabling Printing '''
    sys.stdout = sys.__stdout__

class Map:
    '''
    This Map creates a dictionary comprising the triangulation and features of the cells
    It needs two pictures in the folder:
        "Map/elevation_EI.tif"
        "Map/slope_EI.tif" 
    on my Computer: in "/home/peter/EasterIslands/Code/Triangulation/elevation_EI.tif"
    '''
    def __init__(self, gridpoints_y, r_T, r_F, r_M_later, angleThreshold =10, N_init_trees = 12e6, verbose = True):
        '''
        Creates a Map
        Input Parameters, 
            gridpoints_y: is the number of gridpoints of the pixels in y direction (usually 50), the x-direction has 1.5 as many 
            angle_Threshold=10: All triangles that have angles below that threshold, are masked out of usage
            N_init_trees = 12e6, This amount of trees is spread according to Tree Probability
            verbose=True, if False. Don't print all the statements     
        '''
        if not verbose: blockPrint()

        self.r_M_later = r_M_later
        self.r_T = r_T
        self.r_F = r_F

        EI_elevation = plt.imread("Map/elevation_EI.tif")
        print("Elevation: ", np.min(EI_elevation), np.max(EI_elevation), EI_elevation.shape)
        EI_slope = plt.imread("Map/slope_EI.tif")
        print("Slope: ", np.min(EI_slope), np.max(EI_slope), EI_slope.shape)

        # Create Grid at the pixel points
        self.pixel_dim = EI_elevation.shape
        self.dims_coordinates=[[],[]]
        #self.d_km_pix = 0; self.d_km_pixX = 0
        self.gridpoints_y = gridpoints_y
        self.create_grid_of_points(self.gridpoints_y, EI_elevation)


        # Triangulation
        triObject = mpl.tri.Triangulation(self.points_EI_km[:,0], self.points_EI_km[:,1])
        #self.all_triangles = triObject.triangles
        self.all_triangles = triObject.triangles
        # Mask out unwanted triangles of the triangulation
        self.mask = self.get_mask(angleThreshold, EI_elevation)# self.midpoints_all_int)
        print("Masked out triangles: ", np.sum(self.mask))
        triObject.set_mask(self.mask)
        self.EI_triangles = triObject.get_masked_triangles()
        self.N_c = self.EI_triangles.shape[0]
        print("nr of triangles (after masking): ", self.N_c, " (before): ", self.all_triangles.shape[0])

        # Get Properties of the EI_triangles: midpoint, elevation, slope, area
        midpoints_EI_int = [self.midpoint_int(t) for t in self.EI_triangles]
        self.vec_c = np.array([self.transform(m) for m in midpoints_EI_int])

        # Matrix that calculates all the different points
        distMatrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(self.vec_c)).astype(np.float32)


        elev = np.array([EI_elevation[int(m[1]), int(m[0])] for m in midpoints_EI_int])
        self.el_c = elev.astype(float)*500/255 # 500 is max from Google Earth Engine
        slope = np.array([EI_slope[int(m[1]), int(m[0])] for m in midpoints_EI_int])
        self.sl_c = slope.astype(float) * 30/255 # 30 is max slope from Google Earth Engine
        
        self.A_c = np.array([self.calc_triangle_area(t) for t in self.EI_triangles])
        print("Area of discretised Map is [km2] ", '%.4f' % (np.sum(self.A_c)))
        print("Average triangle area [km2] is ", (np.mean(self.A_c))," +- ", (np.var(self.A_c)**0.5))
        print("  i.e. a triangle edge is on average long: ", (2*(np.mean(self.A_c)))**0.5)# for plotting
        print("Average triangle area [acres] is ", (np.mean(self.A_c * config.km2_to_acre))," +- ", (np.var(self.A_c* config.km2_to_acre)**0.5))
        self.corners = {'upper_left':self.transform([0,self.pixel_dim[0]]),
                        'upper_right':self.transform([self.pixel_dim[1],self.pixel_dim[0]]),
                        'lower_left':self.transform([0,0]),
                        'lower_right':self.transform([self.pixel_dim[1], 1])
                        }
        print("Corners: ", (self.corners))

        # Function to find triangle index of a point
        # self.triangle_finder = triObject.get_trifinder()

        ###########
        # Water

        self.water_triangle_inds = []
        self.water_midpoints = []
        self.P_W = np.array([])
        self.waterpoints(triObject, distMatrix, Lakes=["Raraku", "Kau", "Aroi"])
        self.P_W_NoDrought = copy.deepcopy(self.P_W)
        self.water_triangle_inds_NoDrought= copy.deepcopy(self.water_triangle_inds)
        self.water_midpoints_NoDrought = copy.deepcopy(self.water_midpoints)

        self.water_triangle_inds = []
        self.water_midpoints = []
        self.P_W = np.array([])
        self.waterpoints(triObject, distMatrix, Lakes=["Kau", "Aroi"])
        self.P_W_RarakuDrought = copy.deepcopy(self.P_W)
        self.water_triangle_inds_RarakuDrought= copy.deepcopy(self.water_triangle_inds)
        self.water_midpoints_RarakuDrought= copy.deepcopy(self.water_midpoints)

        ########
        self.P_G = np.array([])
        #self.slopes_cond = np.array([])
        ###########
        # Trees
        # Tree Probability according to condition set in ### config.py ###
        self.treeProb = self.get_TreeProb()
        print("Check if Tree probability works: ", np.sum(self.treeProb), " Cells: ", np.sum(self.treeProb>0), " of ", self.treeProb.shape[0])
        
        # mutable Variables of triangles 
        self.A_F_c = np.zeros([self.N_c], dtype=np.int16) # occupied sites
        self.agNr_c = np.zeros([self.N_c], dtype = np.int16)
        self.pop_c = np.zeros([self.N_c], dtype = np.int16)
        self.T_c= np.zeros([self.N_c], dtype=np.int32)
        
        print("Not Initialising Trees here but later!!")
        #self.init_trees(N_init_trees)
        #print("Total Trees:", np.sum(self.T_c))
        #print("Fraction of islands with forest: ", 
        #    np.sum([self.A_c[i] for i in range(self.N_c) if self.T_c[i]>0])/
        #    (np.sum(self.A_c)))
        self.carrying_cap = np.copy(self.T_c)
        #self.max_T_c = np.max(self.carrying_cap/self.A_c)


        print("Calculate the neighbours for Trees ...")#, end="")
        #self.TreeNeighbours_of_triangles = []
        #self.TreeNeighbourhoodRad = config.params['r_T']
        #self.get_neighbour_triangles_ind(self.TreeNeighbours_of_triangles , self.TreeNeighbourhoodRad)
        #self.TreeNeighbours_of_triangles = np.array(self.TreeNeighbours_of_triangles).astype(int)
        self.C_T_c = np.array(distMatrix<self.r_T, dtype=bool)

        print("Calculate the neighbours for agriculture...")
        #self.AgricNeighbours_of_triangles = []
        #self.AgricNeighbourhoodRad = config.params['r_F']
        #self.get_neighbour_triangles_ind(self.AgricNeighbours_of_triangles , self.AgricNeighbourhoodRad)
        #self.AgricNeighbours_of_triangles = np.array(self.AgricNeighbours_of_triangles).astype(int)
        self.C_F_c = np.array(distMatrix<self.r_F, dtype=bool)

        print("Calculate neighbours for moving (late)")
        self.C_Mlater_c = np.array(distMatrix<self.r_M_later, dtype=bool)

        print("DONE")
        print("Get Agriculture Sites ...")
        self.A_acres_c = np.zeros(self.N_c, dtype=int)
        #self.nr_lowqualitysites = np.zeros(self.N_c)
        self.F_pi_c = np.zeros(self.N_c)
        #self.barrenYears = np.zeros(self.N_c).astype(np.uint8)
        self.well_suited_cells = np.zeros(self.N_c).astype(np.uint8)
        self.poorly_suited_cells = np.zeros(self.N_c).astype(np.uint8)
        self.eroded_well_suited_cells = np.zeros(self.N_c).astype(np.uint8)

        self.get_agriculture_sites(triObject)


        self.get_AnakenaBeach(triObject)
        self.C_F_cAnakena = np.where(distMatrix[:,self.AnakenaBeach_ind]<self.r_F)[0]
        #self.plot_agriculture(which="high", save=True)
        #self.plot_agriculture(which="low", save=True)
        #self.plot_agriculture(which="both", save=True)
        #self.save_ncdf(gridpoints_y)
        return 

    ##############################################
    #####   CREATE THE EI MAP AND TRIANGLES    ###
    ##############################################

    def create_grid_of_points(self, gridpoints_y, EI_elevation):
        ####    Get Dimensions     ####
        pixel_y, pixel_x = self.pixel_dim
        # var geometry = ee.Geometry.Rectangle([-109.465,-27.205, -109.2227, -27.0437]);
        # These coordinates are taken from the Google Earth Engine Bounding Box of the Image
        lonmin, latmin, lonmax, latmax = [-109.465,-27.205, -109.2227, -27.0437]
        #lons = np.linspace(lonmin,lonmax, num=EI_elevation.shape[1])
        #lats = np.linspace(latmin, latmax, num=EI_elevation.shape[0])
        self.dims_coordinates = [ (lonmin, lonmax) ,(latmin, latmax) ]
        print("Dimensions are ", self.dims_coordinates)
        # transform pixels to km
        d_gradlat_per_pixel = abs(latmax-latmin)/pixel_y  #[degrees lat per pixel]
        self.d_km_pixY = 111.320*d_gradlat_per_pixel # [m/lat_deg * lat_deg/pixel = m/pixel]
        # according to wikipedia 1deg Latitude = 111.32km
        d_gradlon_per_pixel = abs(lonmax-lonmin)/pixel_x #[degrees lon per pixel]
        self.d_km_pixX = 111.320 * abs(np.cos((latmax+latmin)*0.5*2*np.pi/360)) * d_gradlon_per_pixel  #[m/pixel]]
        print("1 pixel vertically is [degree lat/pixel]:", '%.8f' % d_gradlat_per_pixel, 
            ", horizontally is [degree lon/pixel]:", '%.8f' % d_gradlon_per_pixel)
        print("km per pixel vertical: ", '%.4f' % self.d_km_pixY)
        print("km per pixel horizontal: ", '%.4f' % self.d_km_pixX)
        print("Box is [km vertically, km horizontally] ", '%.4f' % (self.d_km_pixY * pixel_y ), '%.4f' % (self.d_km_pixX * pixel_x ))
        
        ####    Create grid points ####
        gridpoints_x = int(gridpoints_y * 1.5)

        grid_x = np.linspace(0,pixel_x, num=gridpoints_x, endpoint=False)
        grid_y = np.linspace(0,pixel_y, num=gridpoints_y, endpoint=False)
        
        X_grid,Y_grid = np.meshgrid(grid_x, grid_y)
        x_arr_grid = X_grid.flatten()
        y_arr_grid = Y_grid.flatten()
        # All the pixels of the rectangular picture
        all_points = [[x_arr_grid[k], y_arr_grid[k]]  for k in range(len(x_arr_grid))]

        #Points on Easter Island in pixel coordinates
        self.points_EI_int = np.array([
            [int(p[0]), int(p[1])] for p in all_points if EI_elevation[int(p[1]), int(p[0])] >0
        ], dtype=int)
        print("Nr of cornerpoints on EI: ", self.points_EI_int.shape[0]," of total gridpoints ", len(all_points))
        
        #Points on Easter Island in kms from left lower corner
        self.points_EI_km = np.array([self.transform(p) for p in self.points_EI_int])        
        return 

    def midpoint_int(self, t):
        ''' returns the midpoint in pixel coordinates of a triangle t (corner indices)'''
        return (np.sum(self.points_EI_int[t.astype(int)], axis=0)/3.0)

    def check_angles(self, t, AngleThreshold):
        ''' check whether one of the angles is smaller than the threshold'''
        a,b,c = t
        A, B, C = [self.points_EI_km[k] for k in [a,b,c]]
        cosA = np.dot((B-A), (C-A)) /np.linalg.norm(B-A)/np.linalg.norm(C-A)
        if  cosA<-1 or cosA>1:
            print("Note: An Angle gave the Wrong Value (numerical issues): ", cosA, " at triangle ",  A,B,C, "reduced to [-1,1]")
            if cosA<-1:
                cosA=-1
            elif cosA>1:
                cosA=1
        if cosA>1: cosA=1.0
        if cosA<-1: cosA=-1.0
        angleA = np.arccos(cosA)
        cosB = np.dot((A-B), (C-B)) /np.linalg.norm(A-B)/np.linalg.norm(C-B)
        if cosB>1: cosB=1.0
        if cosB<-1: cosB=-1.0
        angleB = np.arccos(cosB)
        cosC = np.dot((A-C), (B-C)) /np.linalg.norm(A-C)/np.linalg.norm(B-C)
        if cosC>1: cosC=1.0
        if cosC<-1: cosC=-1.0
        angleC = np.arccos(cosC)
        thr = AngleThreshold*2*np.pi/360
        #print(angleA, angleC, angleB, angleA+angleB+angleC)
        return (angleA>thr and angleB>thr and angleC>thr)

    def get_mask(self, AngleThreshold, EI_elevation):
        '''create mask; reduce the all_triangles to EI_triangles'''
        midpoints_all_int = [self.midpoint_int(t) for t in self.all_triangles]
        midpointcond = [(EI_elevation[int(m[1]), int(m[0])]>0) for m in midpoints_all_int]
        anglecond = [self.check_angles(t, AngleThreshold) for t in self.all_triangles]
        mask = [not (midpointcond[k] and anglecond[k]) for k in range(len(anglecond))]
        return mask
        
    def transform(self, p):
        ''' transform pixel coordinates of p into km'''
        return [p[0]*self.d_km_pixX, p[1]*self.d_km_pixY] 
    
    def calc_triangle_area(self, t):
        ''' return the area of a triangle t (indices of corners) in m^2'''
        a,b,c = self.points_EI_km[t]
        ab = b-a; ac = c-a
        return abs(0.5* (ac[0]*ab[1] - ac[1]*ab[0]) )

    ################################################
    ######     GET TRIANGLE FROM POINT     #########
    ################################################

    def get_c_of_point(self, point, triObject):
        ''' returns the triangle of a 2D point given in km coordinates 
        return: index of triangle, midpoint in pixel coord, midpoint in km coordinates
        if point is not in a triangle, which is on Easter Island return ind=-1
        '''
        
        #point = self.transform(point)
        cell_finder = triObject.get_trifinder()
        #mask = triObject.mask
        #all_triangles = triObject.triangles

        ind = cell_finder(point[0], point[1])
        if ind <0 or  triObject.mask[ind]:
            return -1, -1, [-1, -1, -1]
        # The point is on the triangle: found_tri
        found_cell = self.all_triangles[ind]
        # need to get the actual triangle
        m_int = self.midpoint_int(found_cell)
        # Get the index of the triangle in the EI_triangles array, not the all_triangles (including the masked triangles)
        ind = np.where(np.sum((found_cell == self.EI_triangles), axis=1)==3)[0][0]
        return ind, self.transform(m_int) 
    #get_triangle_of_point([300,300]) # in pixel coords, won't work anymore
    #get_triangle_of_point([10000,10000]) #in km

    ############################
    ###    NEIGHBOURS     ######
    ############################

    #def get_neighbour_triangles_ind(self, array, r):
    #    for i in range(self.N_c):
    #        array.append(np.where(self.distMatrix[:,i]<r)[0])

    # OLD
    #def get_neighbour_triangles_ind(self, array, r):
    #    for i in range(self.N_c):
    #        atnb_resourceRad = [int(n) for n in range(self.N_c) 
    #                    if np.linalg.norm(self.vec_c[n]-self.vec_c[i] )<r]
    #        array.append(atnb_resourceRad)
    #    return 

    
    #############################
    #########    TREE     #######
    #############################

    def get_TreeProb(self):
        ''' Function to calculate the tree probability 
        DEPENDENCY OF ELEVATION AND SLOPE TO THE TREE DENSITY
        '''
        minElev = config.TreeDensityConditionParams['minElev']
        ElevHighD = config.TreeDensityConditionParams['ElevHighD']
        maxElev = config.TreeDensityConditionParams['maxElev']
        maxSlope = config.TreeDensityConditionParams['maxSlope']
        maxSlopeHighD = config.TreeDensityConditionParams['maxSlopeHighD']

        factorBetweenHighandLowTreeDensity = config.TreeDensityConditionParams['factorBetweenHighandLowTreeDensity']
        print("Conditions on trees: ", minElev,ElevHighD, maxSlopeHighD, maxElev, maxSlope)
        
        tree_cap = np.zeros([self.N_c])
        for n,_ in enumerate(self.vec_c):
            e = self.el_c[n]
            s = self.sl_c[n]
            elevslope_cond_highDens = ( e<ElevHighD and e>minElev and s<maxSlopeHighD)
            elevslope_cond_lowDens = ( e<maxElev and s<maxSlope)
            water_cond = (not n in self.water_triangle_inds)
            if elevslope_cond_highDens and water_cond:
                tree_cap[n] = factorBetweenHighandLowTreeDensity
            elif elevslope_cond_lowDens and water_cond:
                tree_cap[n] = 1
            else:
                tree_cap[n] = 0
        print("Cells with Trees on them: ", len(np.where(tree_cap==factorBetweenHighandLowTreeDensity)[0]), "(high) and ", 
            len(np.where(tree_cap==1)[0]),"(low) of ", tree_cap.shape[0])
        tree_cap = tree_cap / np.sum(tree_cap)
        return tree_cap

    def init_trees(self, nr_trees):
        ''' initialise nr_trees onto triangles '''
        self.T_c = np.zeros_like(self.T_c)
        inds = np.random.choice(np.arange(self.N_c), 
                            size = int(nr_trees), 
                            p = self.treeProb)
        self.T_c, _ = np.histogram(inds, bins = np.arange(0,self.N_c+1))
        #for i in inds:
        #    self.T_c[i]+=int(1)
        return 

    def fromcoordinate_topixelcoordinate(self, point):
        # point in [-27.bla, -109,...]
        [ (lonmin, lonmax) ,(latmin, latmax) ] = self.dims_coordinates
        # distance in y direction to origin
        pixel_y = np.where(np.linspace(latmin, latmax, num=self.pixel_dim[0]) > point[0])[0][0]
        pixel_x = np.where(np.linspace(lonmin, lonmax, num=self.pixel_dim[1]) > point[1])[0][0]
        return [pixel_x, self.pixel_dim[0]-pixel_y] 


    def get_AnakenaBeach(self, triObject):
        # Pixel Coordinates taken from the elevation.tif (in gimp)
        # lat lon 27°04'23.8"S 109°19'23.6"W, Chile, i.e. (-27.07327778, -109.32305556)
        AnakenaBeach_pixel = self.fromcoordinate_topixelcoordinate((-27.07327778, -109.32305556))
        AnakenaBeach_m =self.transform([AnakenaBeach_pixel[0],AnakenaBeach_pixel[1]]) # This is roughly the coast anakena where they landed.
        
        # get closest midpoint
        #dists = scipy.spatial.distance.cdist(self.vec_c, np.array(AnakenaBeach_m).reshape(1,2))
        self.AnakenaBeach_ind, _ = self.get_c_of_point(AnakenaBeach_m, triObject)
        #self.vec_c[np.argmin(dists)]
        #self.AnakenaBeach_ind, _, _ = self.get_triangle_of_point(AnakenaBeach_m, triObject)

        print("Anakena Beach is at Point:",AnakenaBeach_m, "(",AnakenaBeach_pixel,")", " triangle ", self.AnakenaBeach_ind)
        if self.AnakenaBeach_ind ==-1:      
            print("ERROR, Anakena is not on EI")
            quit()
        return 

    ####################################
    #####    WATER    ##################
    ####################################
        
    def waterpoints(self, triObject, distMatrix, Lakes=["Raraku", "Kau", "Aroi"]):
        ''' get those triangles that contain water, i.e. within the three craters'''
        # from google earth:        
        Raraku = {"midpoint": [-27.121944, -109.2886111], "Radius": 170e-3} # Radius in km
        Kau = {"midpoint": [-27.186111, -109.4352778], "Radius":506e-3} # Radius in km
        Aroi = {"midpoint": [-27.09361111, -109.373888], "Radius":75e-3} # Radius in km
        AllLakeMidpoints = {"Raraku":Raraku['midpoint'], "Kau":Kau['midpoint'], "Aroi":Aroi['midpoint']}
        lake_mid_points = [self.transform(self.fromcoordinate_topixelcoordinate(AllLakeMidpoints[name])) for name in Lakes]
        AllLakeRadius = {"Raraku":Raraku['Radius'], "Kau":Kau['Radius'], "Aroi":Aroi['Radius']}
        lake_radius = [AllLakeRadius[name] for name in Lakes]
        #lake_mid_points = []
        ## this defines the three midpoints (in m) of the three craters.
        #for i,m in enumerate(LakeMidpoints):
        #    lake_midp_kmcoord = self.transform(self.fromcoordinate_topixelcoordinate(m))
        #    lake_mid_points.append(lake_midp_kmcoord)
        self.water_midpoints = []
        self.water_triangle_inds = []
        water_areas = []
        for m,r in zip(lake_mid_points, lake_radius):
            t = self.get_c_of_point(m, triObject)[0]
            self.water_midpoints.append(self.vec_c[t])
            inds_within_lake = np.where((distMatrix[:,t]<r) * (self.sl_c<5))[0]
            if len(inds_within_lake)==0:
                inds_within_lake=[t]
            self.water_triangle_inds.extend(inds_within_lake)
            water_areas.extend([r**2*np.pi for _ in inds_within_lake])
        print("Nr of water triangles: ", len(self.water_triangle_inds))
        print("Sizes of Lakes are [km**2]: ", np.unique(water_areas))
        print("Water Midpoints are at locations (x,y): ", lake_mid_points )

        distance_to_water = distMatrix[self.water_triangle_inds,:]  # shape: water_triangels * N_c
        weighted_squ_distance_to_water = distance_to_water**2/np.array(water_areas)[:,None]  # NOte: Casting to divide each row seperately
        penalties = np.min(weighted_squ_distance_to_water, axis=0)
        #if len(Lakes)==3: self.max_water_penalty_without_drought = np.max(penalties)
        #self.P_W= (penalties/self.max_water_penalty_without_drought ).clip(max=1)
        k_W = config.k_X(config.w01, config.w99)
        self.P_W = config.logistic(penalties, k_W, (config.w01+config.w99)/2)
        
        print("Water Penalties Mean: ", "%.4f"  % (np.mean(self.P_W)))
        return 

    def check_drought(self,t, droughts_RanoRaraku, triObject):
        #drought_RanoRaraku_1 = [config.Starttime, 1100]
        for drought in droughts_RanoRaraku:
            if t==drought[0]:
                print("beginning drought in Raraku, t=",t)
                #self.waterpoints(triObject, Lakes=["Kau", "Aroi"])
                self.water_midpoints= self.water_midpoints_RarakuDrought
                self.P_W = self.P_W_RarakuDrought
                self.water_triangle_inds = self.water_triangle_inds_RarakuDrought
                self.get_P_G()
            if t==drought[1]:
                # end drought:
                #self.waterpoints(triObject, Lakes=["Raraku","Kau", "Aroi"])
                print("ending drought in Raraku, t=",t)
                self.water_midpoints= self.water_midpoints_NoDrought
                self.P_W = self.P_W_NoDrought
                self.water_triangle_inds = self.water_triangle_inds_NoDrought
                self.get_P_G()
        return 



    ################################
    #####     MAP PENALTY   ########
    ################################
    def get_P_G(self):
        k_el = config.k_X(config.el01, config.el99)
        P_el = config.logistic(self.el_c, k_el, (config.el01+config.el99)/2)
        k_sl = config.k_X(config.sl01, config.sl99)
        P_sl = config.logistic(self.sl_c, k_sl, (config.sl01+config.sl99)/2)
        self.P_G = np.maximum(P_el, P_sl)
        self.P_G[self.water_triangle_inds] = 1e6
        return 

    #################################################
    ######       AGRICULTURE                 ########
    #################################################

    def get_agriculture_sites(self, triObject):
        print("Loaded Agriculture Allowance from Puleston 2017.")
        pulestonMap = plt.imread("Map/Agriculture_Puleston2017_scaled.png")
        # The following will encode the colors: Dark Greeen = wellsuited, BrightGreen=poorly suited, Red = non-viable
        data_wellSites = (pulestonMap[:,:,1] > 0.6) * (pulestonMap[:,:,1] <0.9) * (pulestonMap[:,:,2]<0.01)
        data_poorSites = (pulestonMap[:,:,1] >= 0.9)  * (pulestonMap[:,:,2]<0.01)
        
        midpoints_EI_int = [self.midpoint_int(t) for t in self.EI_triangles]

        wellS_allowed = np.array([data_wellSites[int(m[1]), int(m[0])]>0.5 for m in midpoints_EI_int])
        nr_wellsites = wellS_allowed * np.floor(self.A_c*config.km2_to_acre).astype(int)
        poorS_allowed = np.array([data_poorSites[int(m[1]), int(m[0])]>0.5 for m in midpoints_EI_int])
        nr_poorsites = poorS_allowed * np.floor(self.A_c*config.km2_to_acre).astype(int)
 
        print("Well Suited Sites: ", np.sum(nr_wellsites), " on ", np.sum(nr_wellsites>0), " Cells")
        print("Poorly Suited Sites: ", np.sum(nr_poorsites), " on ", np.sum(nr_poorsites>0), " Cells")
        inds_well = np.where(nr_wellsites>0)[0]
        print("Area High Quality: ", np.sum(self.A_c[inds_well]), " (i.e. in Percent: ", np.sum(self.A_c[inds_well]) /np.sum(self.A_c)) 
        inds_poor = np.where(nr_poorsites>0)[0]
        print("Area Low Quality: ", np.sum(self.A_c[inds_poor]), " (i.e. in Percent: ", np.sum(self.A_c[inds_poor]) /np.sum(self.A_c)) 

        self.F_pi_c[inds_well]=config.F_PI_well
        self.F_pi_c[inds_poor]=config.F_PI_poor
        self.well_suited_cells = nr_wellsites
        self.poorly_suited_cells = nr_poorsites

        self.A_acres_c= nr_poorsites+nr_wellsites
        return 


  

    ##############################################
    ##########     PLOTTING    ###################
    ##############################################
    '''
    def plotArea(self, triObject):
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(1,1,1,fc='aquamarine')
        plot = ax.tripcolor(self.points_EI_km[:,0], self.corners['upper_left'][1] - self.points_EI_km[:,1], 
        triObject.get_masked_triangles(), facecolors=self.A_c, vmin = 0, vmax = np.max(self.A_c), cmap="Reds", alpha=1)
        ax.set_aspect('equal')
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        plt.colorbar(plot, cax =cax) 
        plt.savefig("Map/Areas"+str(config.gridpoints_y)+".svg")
        plt.close()
        return 

    def plot_TreeMap_and_hist(self , ax=None, fig=None, name_append = "", hist=False, save=False, histsavename="", folder="Map/"):
        if hist:
            plt.cla()
            plt.figure()
            plt.hist(self.T_c, bins=30)#, range=(0,tree_max))
            plt.xlabel("Number of trees in a triangle on EI")
            plt.ylabel("Frequency")
            plt.savefig(folder+"HistogramInitialTreeNrsPerCell.png")
            plt.close()

        if ax==None:
            fig = plt.figure(figsize=(10,6))
            ax = fig.add_subplot(1,1,1,fc='aquamarine')
        
        #ax.tripcolor(self.points_EI_km[:,0], self.corners['upper_left'][1] - self.points_EI_km[:,1], 
        #    self.EI_triangles, facecolors=np.array([1 for _ in range(self.N_c)]), vmin = 0, vmax = 1, cmap="binary", alpha=1)
        plot = ax.tripcolor(self.points_EI_km[:,0], self.corners['upper_left'][1] - self.points_EI_km[:,1], 
        self.EI_triangles, facecolors=self.T_c, vmin = 0, vmax = np.max(self.carrying_cap), cmap="Greens", alpha=1)
        
        for i in self.water_triangle_inds:
            ax.plot([self.vec_c[i][0]], [self.corners['upper_left'][1]- self.vec_c[i][1]], 'o',markersize=1, color=(3/255,169/255,244/255 ,1 ))

        
        ax.set_aspect('equal')
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        plt.colorbar(plot, cax =cax) 
        if save:
            plt.savefig(folder+"Map_trees"+str(name_append)+"_grid"+str(self.gridpoints_y)+".svg")
            plt.close()
        return fig, ax

    def plot_agricultureSites(self, ax=None, fig=None, save=True, CrossesOrFacecolor="Crosses", folder=""):
        if ax==None:
            fig = plt.figure(figsize=(10,6))
            ax = fig.add_subplot(1,1,1,fc='aquamarine')
            #ax.tripcolor(self.points_EI_km[:,0], self.corners['upper_left'][1] - self.points_EI_km[:,1], 
            #    self.EI_triangles, facecolors=np.array([1 for _ in range(self.N_c)]), vmin = 0, vmax = 1, cmap="binary", alpha=1)

        if CrossesOrFacecolor=="Facecolor":
            plot = ax.tripcolor(self.points_EI_km[:,0], self.corners['upper_left'][1] - self.points_EI_km[:,1], 
                self.EI_triangles, facecolors=self.agriculture, vmin = 0, vmax = np.max(self.agriculture), cmap="Reds", alpha=1)
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", "5%", pad="3%")
            plt.colorbar(plot, cax =cax) 
            #fig.colorbar(plot)#, cax = cbaxes) 
        else:
            inds = np.where(self.agriculture>0)
            ax.plot(self.vec_c[inds,0], self.corners['upper_left'][1]- self.vec_c[inds,1], 'x',markersize=3, color="red")
        
        ax.plot(self.vec_c[self.AnakenaBeach_ind][0],self.corners['upper_left'][1]-self.vec_c[self.AnakenaBeach_ind][1],"x",markersize=5, color="gold" )

        for i in self.water_triangle_inds:
            ax.plot([self.vec_c[i][0]], [self.corners['upper_left'][1]- self.vec_c[i][1]], 'o',markersize=1,color=(3/255,169/255,244/255 ,1 ))
        ax.plot(self.vec_c[self.AnakenaBeach_ind][0],self.corners['upper_left'][1]-self.vec_c[self.AnakenaBeach_ind][1],"x",markersize=5, color="gold" )


        if save:
            plt.savefig(folder+"Map_AgricDensity_grid"+str(self.gridpoints_y)+".svg")
            plt.close()
        return fig, ax

    def plot_water_penalty(self, folder=""):
        plt.cla()
        fig= plt.figure(figsize=(10,6))
        ax = fig.add_subplot(1,1,1,fc='lightgray')
        facecolors = self.P_W
        #ax.tripcolor(self.points_EI_km[:,0], self.corners['upper_left'][1] - self.points_EI_km[:,1], 
        #        self.EI_triangles, facecolors=np.array([1 for _ in range(self.N_c)]), vmin = 0, vmax = 1, cmap="binary", alpha=1)
        plot = ax.tripcolor(self.points_EI_km[:,0], self.corners['upper_left'][1]-self.points_EI_km[:,1], 
                    self.EI_triangles, facecolors=facecolors, vmin = 0, vmax = np.max(self.P_W), cmap="Blues_r")
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        plt.colorbar(plot, cax =cax) 
        for i in self.water_triangle_inds:
            ax.plot([self.vec_c[i][0]], [self.corners['upper_left'][1]- self.vec_c[i][1]], 'o',markersize=1,color="k")
        #plt.savefig(folder+"Map_waterPenalty_grid"+str(self.gridpoints_y)+".svg")
        plt.savefig("Map/Map_waterPenalty_grid"+str(self.gridpoints_y)+".svg")
        plt.close()

    def plot_agriculture(self, which,  ax =None, fig=None, save=False):
        if ax==None:
            fig = plt.figure(figsize=(10,6))
            ax = fig.add_subplot(1,1,1,fc=(3/255,169/255,244/255 ,1 ))
        #ax.tripcolor(self.points_EI_km[:,0], self.corners['upper_left'][1] - self.points_EI_km[:,1], 
        #    self.EI_triangles, facecolors=np.array([1 for _ in range(self.N_c)]), vmin = 0, vmax = 1, cmap="binary", alpha=1)
        if which=="high":
            plot = ax.tripcolor(self.points_EI_km[:,0], self.corners['upper_left'][1] - self.points_EI_km[:,1], 
            self.EI_triangles, facecolors=self.nr_highqualitysites, vmin = 0, vmax = 1#np.max(self.nr_highqualitysites)
            , cmap="Greens", alpha=0.7)
        
        elif which=="low":
            plot = ax.tripcolor(self.points_EI_km[:,0], self.corners['upper_left'][1] - self.points_EI_km[:,1], 
        self.EI_triangles, facecolors=self.nr_lowqualitysites, vmin = 0, vmax = np.max(self.nr_lowqualitysites), cmap="Blues", alpha=0.8)
        elif which =="both":
            plot = ax.tripcolor(self.points_EI_km[:,0], self.corners['upper_left'][1] - self.points_EI_km[:,1], 
        self.EI_triangles, facecolors=1*(self.nr_lowqualitysites>0)+2*(self.nr_highqualitysites>0), vmin = 0, vmax = 2#np.max(self.nr_lowqualitysites+self.nr_highqualitysites)
            , cmap="Reds", alpha=0.8)

        ax.plot(self.vec_c[self.AnakenaBeach_ind][0],self.corners['upper_left'][1]-self.vec_c[self.AnakenaBeach_ind][1],"x",markersize=5, color="gold" )

    
        for i in self.water_triangle_inds:
            ax.plot([self.vec_c[i][0]], [self.corners['upper_left'][1]- self.vec_c[i][1]], 'o',markersize=1,color=(3/255,169/255,244/255 ,1 ))
    
        #cbaxes = fig.add_axes([0.9, 0.0, 0.03, 1.0]) 
        if save:
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", "5%", pad="3%")
            plt.colorbar(plot, cax =cax) 
            plt.savefig("Map/Map_Agriculutre_"+which+"_grid"+str(self.gridpoints_y)+".svg")
        return
    '''

    

                
if __name__=="__main__":

    '''
    config.TreeDensityConditionParams = {'minElev':10, 'maxElev':430,'maxSlope':7}

    config.N_init_trees = 12e6

    config.gridpoints_y=20


    i = Map(config.gridpoints_y, N_init_trees=config.N_init_trees)

    triObject = mpl.tri.Triangulation(config.EI.points_EI_km[:,0], config.EI.points_EI_km[:,1], triangles = i.all_triangles, mask=i.mask)

    i.plot_TreeMap_and_hist(name_append="12e6_e>0", hist=True, save=True)
    #plt.show()
    print(i.get_triangle_of_point([1e4, 1e4], triObject))

    i.plot_water_dist()
    '''
    config.N_trees_arrival = 16e6

    config.gridpoints_y=50
    TreePattern_evenly = {
        'minElev':0, 'maxSlopeHighD': 10, "ElevHighD":450, 
        'maxElev': 450,'maxSlope':10,
        "factorBetweenHighandLowTreeDensity":1}
    TreePattern_twoLevel = {
        'minElev':0, 'maxSlopeHighD': 5, "ElevHighD":250, 
        'maxElev': 450,'maxSlope':10,
        "factorBetweenHighandLowTreeDensity":2}
    config.TreeDensityConditionParams = TreePattern_evenly   

    config.AngleThreshold = 35
    
    config.r_T = 2
    config.r_F = 1
    config.r_M_later=5
    

    config.EI = Map(config.gridpoints_y,config.r_T, config.r_F, config.r_M_later,N_init_trees = config.N_trees_arrival, angleThreshold=config.AngleThreshold)
    filename = "Map/EI_grid"+str(config.gridpoints_y)+"_rad"+str(config.r_T)+"+"+str(config.r_F)+"+"+str(config.r_M_later)

    with open(filename, "wb") as EIfile:
        pickle.dump(config.EI, EIfile)

    config.EI_triObject = mpl.tri.Triangulation(config.EI.points_EI_km[:,0], config.EI.points_EI_km[:,1], triangles = config.EI.all_triangles, mask=config.EI.mask)

    print("Initialising Trees, ...")
    config.N_trees_arrival = 4e6
    config.EI.init_trees(config.N_trees_arrival)
    print("Total Trees:", np.sum(config.EI.T_c))
    print("Fraction of islands with forest: ", 
        np.sum([config.EI.A_c[i] for i in range(config.EI.N_c) if config.EI.T_c[i]>0])/
        (np.sum(config.EI.A_c)))
    # Plot the resulting Map.
    print("Test Map: ",config.EI.get_c_of_point([1e1, 1e1], config.EI_triObject))
    #config.EI.plot_agriculture(which="high", save=True)
    #config.EI.plot_agriculture(which="low", save=True)
    #config.EI.plot_agriculture(which="both", save=True)
    #config.EI.plot_TreeMap_and_hist(name_append="T12e6", hist=True, save=True)
    #config.EI.plot_water_penalty()

    from LogisticRegrowth import regrow_update, popuptrees
    config.tree_regrowth_rate = 0.05
    config.t_arrival=800
    for t in range(config.t_arrival, 1000):
        regrow_update(config.tree_regrowth_rate)
        popuptrees(800)
        print("t", t, ": T(t)=","%.3f" % (np.sum(config.EI.T_c)/1e6), ", PopUp=", config.TreesPoppedUp[-1], ", Regrown=", config.RegrownTrees[-1])