#!/usr/bin/env python3

'''
Peter Steiglechner, 6.3.2020
# edited to fit Model Class on 9.3.2020

This script contains a class Map
It sets up a Map which performs the triangulation, 
    stores triangles (with midpoints, elevation, slope, ...) 
    and variables (tree density, carrying_cap, agriculture)

It uses two pictures of pixel size 900x600 of elevation and slope of Easter Island 
These were created through Google Earth Enginge API
'''

import config
'''
config.py is the file that stores global variables such as EI, map_type (="EI"), 
'''

import matplotlib as mpl 
import numpy as np
import matplotlib.pyplot as plt
#font = {'font.size':20}
#plt.rcParams.update(font)

import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


class Map:
    '''
    This Map creates a dictionary comprising the triangulation and features of the cells
    It needs two pictures in the folder:
        "Map/elevation_EI.tif"
        "Map/slope_EI.tif" 
    on my Computer: in "/home/peter/EasterIslands/Code/Triangulation/elevation_EI.tif"
    '''
    def __init__(self, gridpoints_y, angleThreshold = 10, N_init_trees = 12e6, verbose = True):
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
        #self.d_meter_lat = 0; self.d_meter_lon = 0
        self.create_grid_of_points(gridpoints_y, EI_elevation)

        # Triangulation
        self.all_triangles = mpl.tri.Triangulation(self.points_EI_m[:,0], self.points_EI_m[:,1])

        # Mask out unwanted triangles of the triangulation
        self.mask = self.get_mask(angleThreshold, EI_elevation)# self.midpoints_all_int)
        print("Masked out triangles: ", np.sum(self.mask))
        self.all_triangles.set_mask(self.mask)
        self.EI_triangles = self.all_triangles.get_masked_triangles()
        self.N_els = self.EI_triangles.shape[0]
        print("nr of triangles (after masking): ", self.N_els, " (before): ", self.all_triangles.triangles.shape[0])

        # Get Properties of the EI_triangles: midpoint, elevation, slope, area
        midpoints_EI_int = [self.midpoint_int(t) for t in self.EI_triangles]
        self.EI_midpoints = np.array([self.transform(m) for m in midpoints_EI_int])

        elev = np.array([EI_elevation[int(m[1]), int(m[0])] for m in midpoints_EI_int])
        self.EI_midpoints_elev = elev.astype(float)*500/255 # 500 is max from Google Earth Engine
        slope = np.array([EI_slope[int(m[1]), int(m[0])] for m in midpoints_EI_int])
        self.EI_midpoints_slope = slope.astype(float) * 30/255 # 30 is max slope from Google Earth Engine
        
        self.EI_triangles_areas = np.array([self.calc_triangle_area(t) for t in self.EI_triangles])
        print("Area of discretised Map is [km2] ", '%.4f' % (np.sum(self.EI_triangles_areas)*1e-6))
        print("Average triangle area is ", int(np.mean(self.EI_triangles_areas))," +- ", int(np.var(self.EI_triangles_areas)**0.5))
        # for plotting
        self.corners = {'upper_left':self.transform([0,self.pixel_dim[0]]),
                        'upper_right':self.transform([self.pixel_dim[1],self.pixel_dim[0]]),
                        'lower_left':self.transform([0,0]),
                        'lower_right':self.transform([self.pixel_dim[1], 1])
                        }
        print("Corners: ", (self.corners))

        # Function to find triangle index of a point
        self.triangle_finder = self.all_triangles.get_trifinder()

        # Trees:
        # Tree Probability according to condition set in ### config.py ### TODO
        self.treeProb = self.get_TreeProb()
        print("Check if Tree probability works: ", np.sum(self.treeProb), " Cells: ", np.sum(self.treeProb>0), " of ", self.treeProb.shape[0])
        
        # mutable Variables of triangles 
        self.agriculture = np.zeros([self.N_els], dtype=int)
        self.agentOccupancy = np.zeros([self.N_els], dtype = int)
        self.tree_density= np.zeros([self.N_els], dtype=int)
        
        self.init_trees(N_init_trees)
        print("Total Trees:", np.sum(self.tree_density))

        self.carrying_cap = np.copy(self.tree_density)
        return 


    def create_grid_of_points(self, gridpoints_y, EI_elevation):
        ###############################
        ####    Get Dimensions     ####
        ###############################
        pixel_y, pixel_x = self.pixel_dim
        # var geometry = ee.Geometry.Rectangle([-109.465,-27.205, -109.2227, -27.0437]);
        # These coordinates are taken from the Google Earth Engine Bounding Box of the Image
        lonmin, latmin, lonmax, latmax = [-109.465,-27.205, -109.2227, -27.0437]
        #lons = np.linspace(lonmin,lonmax, num=EI_elevation.shape[1])
        #lats = np.linspace(latmin, latmax, num=EI_elevation.shape[0])
        dims = [ (lonmin, lonmax) ,(latmin, latmax) ]
        print("Dimensions are ", dims)
        # transform pixels to meters
        d_gradlat_per_pixel = abs(latmax-latmin)/pixel_y  #[degrees lat per pixel]
        self.d_meter_lat = 111320*d_gradlat_per_pixel # [m/lat_deg * lat_deg/pixel = m/pixel]
        # according to wikipedia 1deg Latitude = 111.32km
        d_gradlon_per_pixel = abs(lonmax-lonmin)/pixel_x #[degrees lon per pixel]
        self.d_meter_lon = 111320 * abs(np.cos((latmax+latmin)*0.5*2*np.pi/360)) * d_gradlon_per_pixel  #[m/pixel]]
        print("1 pixel up is in degree lat/pixel:", '%.8f' % d_gradlat_per_pixel, 
            ", horizontally in degree lon/pixel:", '%.8f' % d_gradlon_per_pixel)
        print("Meter per pixel vertical: ", '%.4f' % self.d_meter_lon)
        print("Meter per pixel horizontal: ", '%.4f' % self.d_meter_lat)
        print("Picture [m]: ", '%.2f' % (pixel_x*self.d_meter_lat), "x" ,'%.2f' %  (pixel_y*self.d_meter_lon))
        
        ###############################
        ####    Create grid points ####
        ###############################
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
        
        #Points on Easter Island in meters from left lower corner
        self.points_EI_m = np.array([self.transform(p) for p in self.points_EI_int])

        return 

    def midpoint_int(self, t):
        ''' returns the midpoint in pixel coordinates of a triangle t (corner indices)'''
        return (np.sum(self.points_EI_int[t.astype(int)], axis=0)/3.0)

    def check_angles(self, t, AngleThreshold):
        ''' check whether one of the angles is smaller than the threshold'''
        a,b,c = t
        A, B, C = [self.points_EI_m[k] for k in [a,b,c]]
        angleA = np.arccos(np.dot((B-A), (C-A)) /np.linalg.norm(B-A)/np.linalg.norm(C-A))
        angleB = np.arccos(np.dot((A-B), (C-B)) /np.linalg.norm(A-B)/np.linalg.norm(C-B))
        angleC = np.arccos(np.dot((A-C), (B-C)) /np.linalg.norm(A-C)/np.linalg.norm(B-C))
        thr = AngleThreshold*2*np.pi/360
        #print(angleA, angleC, angleB, angleA+angleB+angleC)
        return (angleA>thr and angleB>thr and angleC>thr)

    def get_mask(self, AngleThreshold, EI_elevation):
        '''
        create mask; reduce the self.all_triangles to EI_triangles
        '''
        midpoints_all_int = [self.midpoint_int(t) for t in self.all_triangles.triangles]
        midpointcond = [(EI_elevation[int(m[1]), int(m[0])]>0) for m in midpoints_all_int]
        anglecond = [self.check_angles(t, AngleThreshold) for t in self.all_triangles.triangles]
        mask = [not (midpointcond[k] and anglecond[k]) for k in range(len(anglecond))]
        return mask
        

    def transform(self, p):
        ''' transform pixel coordinates of p into meter'''
        return [p[0]*self.d_meter_lon, p[1]*self.d_meter_lat] 
    
    def calc_triangle_area(self, t):
        ''' return the area of a triangle t (indices of corners) in m^2'''
        a,b,c = self.points_EI_m[t]
        ab = b-a; ac = c-a
        return abs(0.5* (ac[0]*ab[1] - ac[1]*ab[0]) )
        
    def get_triangle_of_point(self, point):
        ''' 
        returns the triangle of a 2D point given in meter coordinates
        return: index of triangle, midpoint in pixel coord, midpoint in meter coordinates
        if point is not in a triangle, which is on Easter Island return ind=-1
        '''
        #point = self.transform(point)
        ind = self.triangle_finder(point[0], point[1])
        if ind <0 or self.all_triangles.mask[ind]:
            return -1, -1, [-1, -1, -1]
        # The point is on the triangle: found_tri
        found_tri = self.all_triangles.triangles[ind]
        # need to get the actual triangle
        m_int = self.midpoint_int(found_tri)
        # Get the index of the triangle in the EI_triangles array, not the all_triangles (including the masked triangles)
        ind = np.where(np.sum((found_tri == self.EI_triangles), axis=1)==3)[0][0]
        return ind, m_int, self.transform(m_int) 
    #get_triangle_of_point([300,300]) # in pixel coords, won't work anymore
    #get_triangle_of_point([10000,10000]) #in meter


    def get_TreeProb(self):
        ''' 
        CRUCIAL FUNCTION THAT CALCULATES THE DEPENDENCY OF ELEVATION AND SLOPE TO THE TREE DENSITY
        '''
        minElev = config.TreeDensityConditionParams['minElev']
        maxElev = config.TreeDensityConditionParams['maxElev']
        maxSlope = config.TreeDensityConditionParams['maxSlope']
        print("Conditions on trees: ", minElev, maxElev, maxSlope)
        
        tree_cap = np.zeros([self.N_els])
        for n,_ in enumerate(self.EI_midpoints):
            e = self.EI_midpoints_elev[n]
            s = self.EI_midpoints_slope[n]
            if e<maxElev and e>minElev and s<maxSlope:
                tree_cap[n] = 1
            else:
                tree_cap[n] = 0
        print("Cells with Trees on them: ", np.sum(tree_cap), " of ", tree_cap.shape[0])
        tree_cap = tree_cap / np.sum(tree_cap)
        return tree_cap

    def init_trees(self, nr_trees):
        ''' initialise nr_trees onto triangles '''
        self.tree_density = np.zeros_like(self.tree_density)
        inds = np.random.choice(np.arange(self.N_els), 
                            size = int(nr_trees), 
                            p = self.treeProb)
        for i in inds:
            self.tree_density[i]+=int(1)
        return 

    def plot_TreeMap_and_hist(self , name_append = ""):
        plt.cla()
        plt.hist(self.tree_density, bins=30)#, range=(0,tree_max))
        plt.savefig("Histo_test"+str(name_append)+".png")
        plt.close()
        plt.figure(figsize=(10,6))
        plt.tripcolor(self.points_EI_m[:,0], self.corners['upper_left'][1]-self.points_EI_m[:,1], 
            self.EI_triangles, facecolors=self.tree_density, vmin = 0, vmax = np.max(self.carrying_cap), cmap="Greens")
        plt.colorbar()
        plt.savefig("Map_trees_test"+str(name_append)+".png")
        plt.close()

    
        

                
if __name__=="__main__":

    i = Map(50, N_init_trees=1e6)
    i.plot_TreeMap_and_hist(name_append="1e6_e>0")

    print(i.get_triangle_of_point([1e4, 1e4]))