#!/usr/bin/env python3


#Peter Steiglechner, 6.3.2020
## edited to fit Model Class on 9.3.2020
## edited to include a water penalty
## edited to be able to pickle.dump a object.
## edited to give nr of agriculture sites 25.03.2020
## edited to get agriculture not from elevation/slope but from Paper Puleston2017  02.04.2020
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
    def __init__(self, gridpoints_y, angleThreshold =10, N_init_trees = 12e6, verbose = True):
        '''
        Creates a Map
        Input Parameters, 
            gridpoints_y: is the number of gridpoints of the pixels in y direction (usually 50), the x-direction has 1.5 as many 
            angle_Threshold=10: All triangles that have angles below that threshold, are masked out of usage
            N_init_trees = 12e6, This amount of trees is spread according to Tree Probability
            verbose=True, if False. Don't print all the statements     
        '''
        if not verbose: blockPrint()

        EI_elevation = plt.imread("Map/elevation_EI.tif")
        print("Elevation: ", np.min(EI_elevation), np.max(EI_elevation), EI_elevation.shape)
        EI_slope = plt.imread("Map/slope_EI.tif")
        print("Slope: ", np.min(EI_slope), np.max(EI_slope), EI_slope.shape)

        # Create Grid at the pixel points
        self.pixel_dim = EI_elevation.shape
        self.dims_coordinates=[[],[]]
        #self.d_km_lat = 0; self.d_km_lon = 0
        self.gridpoints_y = gridpoints_y
        self.create_grid_of_points(self.gridpoints_y, EI_elevation)

        # Triangulation
        triObject = mpl.tri.Triangulation(self.points_EI_km[:,0], self.points_EI_km[:,1])
        self.all_triangles = triObject.triangles
        # Mask out unwanted triangles of the triangulation
        self.mask = self.get_mask(angleThreshold, EI_elevation)# self.midpoints_all_int)
        print("Masked out triangles: ", np.sum(self.mask))
        triObject.set_mask(self.mask)
        self.EI_triangles = triObject.get_masked_triangles()
        self.N_els = self.EI_triangles.shape[0]
        print("nr of triangles (after masking): ", self.N_els, " (before): ", self.all_triangles.shape[0])

        # Get Properties of the EI_triangles: midpoint, elevation, slope, area
        midpoints_EI_int = [self.midpoint_int(t) for t in self.EI_triangles]
        self.EI_midpoints = np.array([self.transform(m) for m in midpoints_EI_int])

        elev = np.array([EI_elevation[int(m[1]), int(m[0])] for m in midpoints_EI_int])
        self.EI_midpoints_elev = elev.astype(float)*500/255 # 500 is max from Google Earth Engine
        slope = np.array([EI_slope[int(m[1]), int(m[0])] for m in midpoints_EI_int])
        self.EI_midpoints_slope = slope.astype(float) * 30/255 # 30 is max slope from Google Earth Engine
        
        self.EI_triangles_areas = np.array([self.calc_triangle_area(t) for t in self.EI_triangles])
        print("Area of discretised Map is [km2] ", '%.4f' % (np.sum(self.EI_triangles_areas)))
        print("Average triangle area [km2] is ", (np.mean(self.EI_triangles_areas))," +- ", (np.var(self.EI_triangles_areas)**0.5))
        print("  i.e. a triangle edge is on average long: ", (2*(np.mean(self.EI_triangles_areas)))**0.5)# for plotting
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
        self.water_penalties = []
        self.waterpoints()
        ########

        ###########
        # Trees
        # Tree Probability according to condition set in ### config.py ###
        self.treeProb = self.get_TreeProb()
        print("Check if Tree probability works: ", np.sum(self.treeProb), " Cells: ", np.sum(self.treeProb>0), " of ", self.treeProb.shape[0])
        
        # mutable Variables of triangles 
        self.agriculture = np.zeros([self.N_els], dtype=np.int32)
        self.agentOccupancy = np.zeros([self.N_els], dtype = np.int32)
        self.populationOccupancy = np.zeros([self.N_els], dtype = np.int32)
        self.tree_density= np.zeros([self.N_els], dtype=np.int32)
        
        self.init_trees(N_init_trees)
        print("Total Trees:", np.sum(self.tree_density))
        print("Fraction of islands with forest: ", 
            np.sum([self.EI_triangles_areas[i] for i in range(self.N_els) if self.tree_density[i]>0])/
            (np.sum(self.EI_triangles_areas)))


        self.carrying_cap = np.copy(self.tree_density)
        self.max_tree_density = np.max(self.carrying_cap/self.EI_triangles_areas)


        print("Calculate the neighbours for Trees ...")#, end="")
        self.TreeNeighbours_of_triangles = []
        self.TreeNeighbourhoodRad = config.params['tree_search_radius']
        self.get_neighbour_triangles_ind(self.TreeNeighbours_of_triangles , self.TreeNeighbourhoodRad)
        #self.TreeNeighbours_of_triangles = np.array(self.TreeNeighbours_of_triangles).astype(int)

        print("Calculate the neighbours for agriculture...")
        self.AgricNeighbours_of_triangles = []
        self.AgricNeighbourhoodRad = config.params['agriculture_radius']
        self.get_neighbour_triangles_ind(self.AgricNeighbours_of_triangles , self.AgricNeighbourhoodRad)
        #self.AgricNeighbours_of_triangles = np.array(self.AgricNeighbours_of_triangles).astype(int)

        print("DONE")

        self.nr_highqualitysites = np.zeros(self.N_els)
        self.nr_lowqualitysites = np.zeros(self.N_els)
        self.get_agriculture_sites()



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
        self.d_km_lat = 111.320*d_gradlat_per_pixel # [m/lat_deg * lat_deg/pixel = m/pixel]
        # according to wikipedia 1deg Latitude = 111.32km
        d_gradlon_per_pixel = abs(lonmax-lonmin)/pixel_x #[degrees lon per pixel]
        self.d_km_lon = 111.320 * abs(np.cos((latmax+latmin)*0.5*2*np.pi/360)) * d_gradlon_per_pixel  #[m/pixel]]
        print("1 pixel up is in degree lat/pixel:", '%.8f' % d_gradlat_per_pixel, 
            ", horizontally in degree lon/pixel:", '%.8f' % d_gradlon_per_pixel)
        print("km per pixel vertical: ", '%.4f' % self.d_km_lon)
        print("km per pixel horizontal: ", '%.4f' % self.d_km_lat)
        print("Picture [m]: ", '%.2f' % (pixel_x*self.d_km_lat), "x" ,'%.2f' %  (pixel_y*self.d_km_lon))
        
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
        ])
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
        if  cosA<=-1 or cosA>=1:
            print("Note: An Angle gave the Wrong Value (numerical issues): ", cosA, " at triangle ",  A,B,C, "reduced to [-1,1]")
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
        return [p[0]*self.d_km_lon, p[1]*self.d_km_lat] 
    
    def calc_triangle_area(self, t):
        ''' return the area of a triangle t (indices of corners) in m^2'''
        a,b,c = self.points_EI_km[t]
        ab = b-a; ac = c-a
        return abs(0.5* (ac[0]*ab[1] - ac[1]*ab[0]) )

    ################################################
    ######     GET TRIANGLE FROM POINT     #########
    ################################################

    def get_triangle_of_point(self, point, triObject):
        ''' returns the triangle of a 2D point given in km coordinates 
        return: index of triangle, midpoint in pixel coord, midpoint in km coordinates
        if point is not in a triangle, which is on Easter Island return ind=-1
        '''
        
        #point = self.transform(point)
        triangle_finder = triObject.get_trifinder()
        mask = triObject.mask
        all_triangles = triObject.triangles

        ind = triangle_finder(point[0], point[1])
        if ind <0 or mask[ind]:
            return -1, -1, [-1, -1, -1]
        # The point is on the triangle: found_tri
        found_tri = all_triangles[ind]
        # need to get the actual triangle
        m_int = self.midpoint_int(found_tri)
        # Get the index of the triangle in the EI_triangles array, not the all_triangles (including the masked triangles)
        ind = np.where(np.sum((found_tri == self.EI_triangles), axis=1)==3)[0][0]
        return ind, m_int, self.transform(m_int) 
    #get_triangle_of_point([300,300]) # in pixel coords, won't work anymore
    #get_triangle_of_point([10000,10000]) #in km

    ############################
    ###    NEIGHBOURS     ######
    ############################

    def get_neighbour_triangles_ind(self, array, r):
        for i in range(self.N_els):
            atnb_resourceRad = [int(n) for n in range(self.N_els) 
                        if np.linalg.norm(self.EI_midpoints[n]-self.EI_midpoints[i] )<r]
            array.append(atnb_resourceRad)
        return 

    
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
        
        tree_cap = np.zeros([self.N_els])
        for n,_ in enumerate(self.EI_midpoints):
            e = self.EI_midpoints_elev[n]
            s = self.EI_midpoints_slope[n]
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
        self.tree_density = np.zeros_like(self.tree_density)
        inds = np.random.choice(np.arange(self.N_els), 
                            size = int(nr_trees), 
                            p = self.treeProb)
        self.tree_density, _ = np.histogram(inds, bins = np.arange(0,self.N_els+1))
        #for i in inds:
        #    self.tree_density[i]+=int(1)
        return 

    def fromcoordinate_topixelcoordinate(self, point):
        # point in [-27.bla, -109,...]
        [ (lonmin, lonmax) ,(latmin, latmax) ] = self.dims_coordinates
        # distance in y direction to origin
        pixel_y = np.where(np.linspace(latmin, latmax, num=self.pixel_dim[0]) > point[0])[0][0]
        pixel_x = np.where(np.linspace(lonmin, lonmax, num=self.pixel_dim[1]) > point[1])[0][0]
        return [pixel_x, self.pixel_dim[0]-pixel_y] 

    ####################################
    #####    WATER    ##################
    ####################################
        
    def waterpoints(self):
        ''' get those triangles that contain water, i.e. within the three craters'''
        # from google earth:        
        Raraku = {"midpoint": [-27.121944, -109.2886111], "Radius": 170e-3} # Radius in km
        Kau = {"midpoint": [-27.186111, -109.4352778], "Radius":506e-3} # Radius in km
        Aroi = {"midpoint": [-27.09361111, -109.373888], "Radius":75e-3} # Radius in km
        lake_mid_points = []
        # this defines the three midpoints (in m) of the three craters.
        for i,m in enumerate([Raraku['midpoint'], Kau['midpoint'], Aroi['midpoint']]):
            lake_midp_kmcoord = self.transform(self.fromcoordinate_topixelcoordinate(m))
            lake_mid_points.append(lake_midp_kmcoord)
        
        # This looks for the distances of the triangles to each of these midpoints
        water_dist_of_triangle = [] # shape: N_triangles, 3 (water sources)
        for i,m in enumerate(self.EI_midpoints):
            dist_to_water = [np.linalg.norm(np.array(m_water)-np.array(m)) for m_water in lake_mid_points]
            water_dist_of_triangle.append(dist_to_water)
        water_dist_of_triangle  = np.array(water_dist_of_triangle)
        
        # Then check, which of the triangles have midpoints in the radius around those Lake_midpoints
        #separated_waters = [[],[],[]]
        self.water_midpoints = []
        whichwater=[]
        water_areas=[]
        for j, r in enumerate([Raraku['Radius'], Kau['Radius'], Aroi['Radius']]):
            inds = np.where([(water_dist_of_triangle[n,j] < r and self.EI_midpoints_slope[n]<3) for n in range(self.N_els)])
            if len(inds[0])==0:
                # None of the midpoints are on the actual lake, anyway, just choose ONE nearest single triangle
                min_ind = np.where(water_dist_of_triangle[:,j] == min(water_dist_of_triangle[:,j]))[0][0]
                self.water_midpoints.append(self.EI_midpoints[min_ind])
                self.water_triangle_inds.append(min_ind)
                whichwater.append(j)
                #separated_waters[j].append(min_ind)
            else:
                # Add all the triangles with midpoints within the radius and add them to the specific lake (j)
                for i in inds[0]:#whichwater = water_dist_of_triangle.index(min(water_dist_of_triangle))
                    self.water_midpoints.append(self.EI_midpoints[i])
                    self.water_triangle_inds.append(i)
                    whichwater.append(j)
                    #separated_waters[j].append(i)
            # add all water of this lake (j) together
            #water_areas.append(np.sum([self.EI_triangles_areas[b] for n,b in enumerate(self.water_triangle_inds) if whichwater[n]==j]))
            water_areas.append(np.pi * r**2)
        print("There are Nr of Water Triangles", len(self.water_midpoints))
        #water_areas=[]
        #for j in range(3):
        #    water_areas.append(np.sum([self.EI_triangles_areas[i] for i in separated_waters[j]]))
        
        print("Water areas: ", water_areas)
        
        # Check again the water distance of each triangle with now all water points not only midpoints.
        # assign penalty 
        for i,m in enumerate(self.EI_midpoints):
            water_dist_of_triangle = []
            weighted_water_dist_of_triangle = []
            for j,m_water in zip(whichwater, self.water_midpoints):
                water_dist_of_triangle.append(np.linalg.norm(np.array(m_water)-np.array(m)))
                weight = (1/water_areas[j])#**0.5
                weighted_water_dist_of_triangle.append(water_dist_of_triangle[-1]**2 * weight)
            self.water_penalties.append(min(weighted_water_dist_of_triangle))
        self.water_penalties= np.array(self.water_penalties)/np.max(self.water_penalties)


    #################################################
    ######       AGRICULTURE                 ########
    #################################################

    def get_agriculture_sites(self):
        print("Loaded Agriculture Allowance from Puleston 2017.")
        pulestonMap = plt.imread("Map/Agriculture_Puleston2017_scaled.png")
        data_highqualitySites = (pulestonMap[:,:,1] > 0.6) * (pulestonMap[:,:,1] <0.9) * (pulestonMap[:,:,2]<0.01)
        data_lowqualitySites = (pulestonMap[:,:,1] >= 0.9)  * (pulestonMap[:,:,2]<0.01)
        
        midpoints_EI_int = [self.midpoint_int(t) for t in self.EI_triangles]

        highqu_allowed = np.array([data_highqualitySites[int(m[1]), int(m[0])]>0.5 for m in midpoints_EI_int])
        self.nr_highqualitysites = highqu_allowed * np.floor(self.EI_triangles_areas*config.km2_to_acre).astype(int)
        lowqu_allowed = np.array([data_lowqualitySites[int(m[1]), int(m[0])]>0.5 for m in midpoints_EI_int])
        self.nr_lowqualitysites = lowqu_allowed * np.floor(self.EI_triangles_areas*config.km2_to_acre).astype(int)
 
        print("High Quality SItes: ", np.sum(self.nr_highqualitysites), " on ", np.sum(self.nr_highqualitysites>0))
        print("Low Quality SItes: ", np.sum(self.nr_lowqualitysites), " on ", np.sum(self.nr_lowqualitysites>0))
        inds_high = np.where(self.nr_highqualitysites>0)[0]
        print("Area High Quality: ", np.sum(self.EI_triangles_areas[inds_high]), " (i.e. in Percent: ", np.sum(self.EI_triangles_areas[inds_high]) /np.sum(self.EI_triangles_areas)) 
        inds_low = np.where(self.nr_lowqualitysites>0)[0]
        print("Area Low Quality: ", np.sum(self.EI_triangles_areas[inds_low]), " (i.e. in Percent: ", np.sum(self.EI_triangles_areas[inds_low]) /np.sum(self.EI_triangles_areas)) 

    
    #  def get_agriculture_sites(self):
    #      ''' Agriculture Site Availability depends on elevation and slope '''
    #      minElev = config.AgriConds['minElev']
    #      maxElev_highQu = config.AgriConds['maxElev_highQu']
    #      maxElev_lowQu = config.AgriConds['maxElev_lowQu']
    #      maxSlope_highQu = config.AgriConds['maxSlope_highQu']
    #      maxSlope_lowQu = config.AgriConds['maxSlope_lowQu']
 
    #      print("Conditions on Agri: ", minElev, maxElev_highQu, maxElev_lowQu, maxSlope_highQu, maxSlope_lowQu)
 
    #      for n,_ in enumerate(self.EI_midpoints):
    #          e = self.EI_midpoints_elev[n]
    #          s = self.EI_midpoints_slope[n]
    #          elevslope_high_cond = ( e<maxElev_highQu and e>minElev and s<maxSlope_highQu)
    #          elevslope_low_cond = (e<maxElev_lowQu and e>minElev) and (s<maxSlope_lowQu)
    #          not_on_water_cond = (not n in self.water_triangle_inds) 
    #          available_water_cond = (self.water_penalties[n] < config.AgriConds['MaxWaterPenalty'])
    #          
    #          if elevslope_high_cond and available_water_cond and not_on_water_cond:
    #              self.nr_highqualitysites[n] =  int(self.EI_triangles_areas[n]*config.km2_to_acre) # Flenley/Bahn 2017 p 218, 2 acres per 5-7 people- 1 hh will need up to 8 agriculture sites.
    #              self.nr_lowqualitysites[n] = 0
    #          else:
    #              self.nr_highqualitysites[n]=0
    #              if elevslope_low_cond and not_on_water_cond:
    #                  self.nr_lowqualitysites[n] = int(self.EI_triangles_areas[n]*config.km2_to_acre/(2)) 
    #              else:
    #                  self.nr_lowqualitysites[n] = 0
 
    #      print("High Quality SItes: ", np.sum(self.nr_highqualitysites), " on ", np.sum(self.nr_highqualitysites>0))
    #      print("Low Quality SItes: ", np.sum(self.nr_lowqualitysites), " on ", np.sum(self.nr_lowqualitysites>0))
    #      inds_high = np.where(self.nr_highqualitysites>0)[0]
    #      print("Area High Quality: ", np.sum(self.EI_triangles_areas[inds_high]), " (i.e. in Percent: ", np.sum(self.EI_triangles_areas[inds_high]) /np.sum(self.EI_triangles_areas)) 
    #      inds_low = np.where(self.nr_lowqualitysites>0)[0]
    #      print("Area Low Quality: ", np.sum(self.EI_triangles_areas[inds_low]), " (i.e. in Percent: ", np.sum(self.EI_triangles_areas[inds_low]) /np.sum(self.EI_triangles_areas)) 
 
    #      return # self.nr_highqualitysites, self.nr_lowqualitysites  
    #  
 

    #################################################
    ######       SAVE NCDF NOT USED          ########
    #################################################

    
    # def save_ncdf(self, gridpoints_y):
    #     ds = xr.Dataset(
    #         {
    #             'tree_density':(("triangle_ind"),self.tree_density),
    #             'EI_midpoints':(("triangle_ind", "xy"),self.EI_midpoints),
    #             'agentOccupancy':(("triangle_ind"),self.agentOccupancy),
    #             'populationOccupancy':(("triangle_ind"),self.populationOccupancy),
    #             'points_EI_km':(("points_ind", "xy"),self.points_EI_km),
    #             'EI_triangles':(("triangle_ind", "tri"),self.EI_triangles),
    #             'carrying_cap':(("triangle_ind"),self.carrying_cap),
    #             'EI_triangles_areas':(("triangle_ind"),self.EI_triangles_areas),
    #             'EI_midpoints_elev':(("triangle_ind"),self.EI_midpoints_elev),
    #             'EI_midpoints_slope':(("triangle_ind"),self.EI_midpoints_slope),
    #             'water_penalties':(("triangle_ind"),self.water_penalties),
    #         }, 
    #         coords = 
    #             {
    #                 "triangles_ind":np.arange(self.N_els),
    #                 "points_ind":np.arange(len(self.points_EI_km)),
    #                 "xy":[0,1],
    #                 "tri":[0,1,2]
    #             }
    #     )
    #     ds.attrs["N_els"] = self.N_els
    #     ds.attrs["corner upper left"] = self.corners['upper_left']
    #     ds.attrs["pixel_dim"]=self.pixel_dim
    #     ds.attrs['d_km_lat']=self.d_km_lat
    #     ds.attrs['d_km_lon']=self.d_km_lon
    #     ds.attrs['gridpoints_y'] = gridpoints_y
    #     ds.to_netcdf("EImap_"+str(gridpoints_y)+".ncdf")
    #     return 
   

    ##############################################
    ##########     PLOTTING    ###################
    ##############################################

    def plot_TreeMap_and_hist(self , ax=None, fig=None, name_append = "", hist=False, save=False, histsavename="", folder="Map/"):
        if hist:
            plt.cla()
            plt.figure()
            plt.hist(self.tree_density, bins=30)#, range=(0,tree_max))
            plt.xlabel("Number of trees in a triangle on EI")
            plt.ylabel("Frequency")
            plt.savefig(folder+"HistogramInitialTreeNrsPerCell.png")
            plt.close()

        if ax==None:
            fig = plt.figure(figsize=(10,6))
            ax = fig.add_subplot(1,1,1,fc='aquamarine')
        
        #ax.tripcolor(self.points_EI_km[:,0], self.corners['upper_left'][1] - self.points_EI_km[:,1], 
        #    self.EI_triangles, facecolors=np.array([1 for _ in range(self.N_els)]), vmin = 0, vmax = 1, cmap="binary", alpha=1)
        plot = ax.tripcolor(self.points_EI_km[:,0], self.corners['upper_left'][1] - self.points_EI_km[:,1], 
        self.EI_triangles, facecolors=self.tree_density, vmin = 0, vmax = np.max(self.carrying_cap), cmap="Greens", alpha=1)
        
        for i in self.water_midpoints:
            ax.plot([i[0]], [self.corners['upper_left'][1]- i[1]], 'o',markersize=1, color="blue")
        
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
            #    self.EI_triangles, facecolors=np.array([1 for _ in range(self.N_els)]), vmin = 0, vmax = 1, cmap="binary", alpha=1)

        if CrossesOrFacecolor=="Facecolor":
            plot = ax.tripcolor(self.points_EI_km[:,0], self.corners['upper_left'][1] - self.points_EI_km[:,1], 
                self.EI_triangles, facecolors=self.agriculture, vmin = 0, vmax = np.max(self.agriculture), cmap="Reds", alpha=1)
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", "5%", pad="3%")
            plt.colorbar(plot, cax =cax) 
            #fig.colorbar(plot)#, cax = cbaxes) 
        else:
            inds = np.where(self.agriculture>0)
            ax.plot(self.EI_midpoints[inds,0], self.corners['upper_left'][1]- self.EI_midpoints[inds,1], 'x',markersize=3, color="red")
        
        for i in self.water_midpoints:
            ax.plot([i[0]], [self.corners['upper_left'][1]- i[1]], 'x',markersize=3, color=(3/255,169/255,244/255 ,1 ))

        if save:
            plt.savefig(folder+"Map_AgricDensity_grid"+str(self.gridpoints_y)+".svg")
            plt.close()
        return fig, ax

    def plot_water_penalty(self, folder=""):
        plt.cla()
        fig= plt.figure(figsize=(10,6))
        ax = fig.add_subplot(1,1,1,fc='lightgray')
        facecolors = self.water_penalties
        #ax.tripcolor(self.points_EI_km[:,0], self.corners['upper_left'][1] - self.points_EI_km[:,1], 
        #        self.EI_triangles, facecolors=np.array([1 for _ in range(self.N_els)]), vmin = 0, vmax = 1, cmap="binary", alpha=1)
        plot = ax.tripcolor(self.points_EI_km[:,0], self.corners['upper_left'][1]-self.points_EI_km[:,1], 
                    self.EI_triangles, facecolors=facecolors, vmin = 0, vmax = np.max(self.water_penalties), cmap="Blues_r")
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        plt.colorbar(plot, cax =cax) 
        for i in self.water_midpoints:
            ax.plot([i[0]], [self.corners['upper_left'][1]- i[1]], 'o',markersize=1, color="red")
        #plt.show()
        #plt.savefig(folder+"Map_waterPenalty_grid"+str(self.gridpoints_y)+".svg")
        plt.savefig("Map/Map_waterPenalty_grid"+str(self.gridpoints_y)+".svg")
        plt.close()

    def plot_agriculture(self, which,  ax =None, fig=None, save=False):
        if ax==None:
            fig = plt.figure(figsize=(10,6))
            ax = fig.add_subplot(1,1,1,fc=(3/255,169/255,244/255 ,1 ))
        #ax.tripcolor(self.points_EI_km[:,0], self.corners['upper_left'][1] - self.points_EI_km[:,1], 
        #    self.EI_triangles, facecolors=np.array([1 for _ in range(self.N_els)]), vmin = 0, vmax = 1, cmap="binary", alpha=1)
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
    
        for i in self.water_midpoints:
            plt.plot([i[0]], [self.corners['upper_left'][1]- i[1]], 'o',markersize=1, color="black")
            
        #cbaxes = fig.add_axes([0.9, 0.0, 0.03, 1.0]) 
        if save:
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", "5%", pad="3%")
            plt.colorbar(plot, cax =cax) 
            plt.savefig("Map/Map_Agriculutre_"+which+"_grid"+str(self.gridpoints_y)+".svg")
        return


    

                
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
    config.N_trees = 12e6

    config.gridpoints_y=100
    config.TreeDensityConditionParams = {'minElev':5, 'maxElev': 430,'maxSlope':9,'maxSlopeHighD': 5, "ElevHighD":250, "factorBetweenHighandLowTreeDensity":2}
    #config.AgriConds={'minElev':5,'maxElev_highQu':120,
    #    'maxSlope_highQu':3,'maxElev_lowQu':300,'maxSlope_lowQu':5,
    #    'MaxWaterPenalty':0.7,}
    # REPLACED BY AGRICULTURE FROM PULESTON 2017
    config.AngleThreshold = 15

    config.params={'resource_search_radius':0.5}

    config.EI = Map(config.gridpoints_y,N_init_trees = config.N_trees, angleThreshold=config.AngleThreshold)
    with open("Map/EI_grid"+str(config.gridpoints_y)+"_rad"+str(config.params['resource_search_radius']), "wb") as EIfile:
        
        pickle.dump(config.EI, EIfile)

    config.EI_triObject = mpl.tri.Triangulation(config.EI.points_EI_km[:,0], config.EI.points_EI_km[:,1], triangles = config.EI.all_triangles, mask=config.EI.mask)

    # Plot the resulting Map.
    print("Test Map: ",config.EI.get_triangle_of_point([1e1, 1e1], config.EI_triObject))
    config.EI.plot_agriculture(which="high", save=True)
    config.EI.plot_agriculture(which="low", save=True)
    config.EI.plot_agriculture(which="both", save=True)
    config.EI.plot_TreeMap_and_hist(name_append="T12e6", hist=True, save=True)
    config.EI.plot_water_penalty()
