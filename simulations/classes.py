# System-level imports
import argparse
import json
import os
import sys
import time

# Used for managing parallel calculations
import multiprocessing
import tqdm

# Packages: Doing math
import numpy as np
import pandas as pd
import math

# Packages: Visualizing data
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.colors import colorConverter
plt.switch_backend('agg')

# Packages: Managing data
import signac

# Packages: Reading data
import gtar
import glotzformats
import glotzformats.reader as reader
import glotzformats.trajectory as tj

# Packages: Analyzing data
import freud
import freud.pmft as pmft
import freud.density as density
import freud.box as box

# Import hoomd
import hoomd
from hoomd import dem

from . import viz as viz
from . import utils as util

def rebox_positions(x_pos,y_pos,L,center):
    x_pos_new = np.copy(x_pos-center[0])
    y_pos_new = np.copy(y_pos-center[1])

    for i in range(x_pos.shape[0]):
        if x_pos_new[i]<=(-L/2): x_pos_new[i]+=L
        if x_pos_new[i]>(L/2): x_pos_new[i]-=L
        if y_pos_new[i]<=(-L/2): y_pos_new[i]+=L
        if y_pos_new[i]>(L/2): y_pos_new[i]-=L

    return x_pos_new, y_pos_new

def rebox_distances(x_diff,y_diff,L):
    for i in range(x_diff.shape[0]):
        if x_diff[i]<=(-L/2): x_diff[i]+=L
        if x_diff[i]>(L/2): x_diff[i]-=L
        if y_diff[i]<=(-L/2): y_diff[i]+=L
        if y_diff[i]>(L/2): y_diff[i]-=L
    return x_diff, y_diff

'''
Connects to MongoDB database and loads queried trajectory files
'''
class SignacFile(object):
    def __init__(self,sp,index=None):
        self.sp = sp
        db = signac.db.get_database('active_polygons')
        db.connect = False
        if index == None: self.index = db.Production_ngons
        elif index =="disks": self.index = db.Control_disks_L
        elif index =="production_disks": self.index = db.Production_disks
        else: raise ValueError('non-valid database index name passed to SignacFile class')
        self.get_query()
        self.get_doc()
        self.update_sp()
        self.get_fileroot()

    def get_query(self):
        sp = self.sp
        query = {}
        try: query['statepoint.packing_fraction'] = {'$eq': sp['packing_fraction']}
        except: pass
        try: query['statepoint.n_gons'] = {'$eq': sp['n_gons']}
        except: pass
        try: query['statepoint.replicate_id'] = {'$eq': sp['replicate']}
        except: pass
        try: query['statepoint.force_offset'] = {'$eq': sp['force_offset']}
        except: pass
        try: query['statepoint.Pe'] = {'$eq': sp['Pe']}
        except: pass
        self.query = query

    def get_doc(self, doc_num=0):
        docs = self.index.find(self.query)
        self.doc = docs[doc_num]

    def update_sp(self):
        self.sp = self.doc['statepoint']

    def get_fileroot(self):
        doc = self.doc
        self.fileroot = doc['root'] + '/' + doc['signac_id']
        if sys.platform=='darwin': # desktop configuration
            self.fileroot = self.fileroot.replace('/turbo/glotzerCBES/active-shapes/moranse', '/glotzerCBES/active-polygons')
        elif sys.platform=='linux': # vislab configuration
            self.fileroot = self.fileroot.replace('/nfs/glotzerCBES/active-shapes/moranse', '/nfs/turbo/glotzerCBES/active-polygons')
            self.fileroot = self.fileroot.replace('/nfs/glotzerCBES/active-shapes', '/nfs/turbo/glotzerCBES/active-polygons')
        else: pass
        self.fileroot = self.fileroot.replace('/active_shape_zoo_open/01_N-gons/00_Runscripts/','/experiments/')

    # Methods below here not called in initialization
    def get_traj(self):
        try:
            return self.traj
        except AttributeError:
            zip_init = self.fileroot +'/init.zip'
            dcd_file = self.fileroot +'/dump.dcd'

            zip_reader = reader.GetarFileReader()
            dcd_reader = reader.DCDFileReader()

            with open(zip_init, 'rb') as initfile:
                with open(dcd_file, 'rb') as dcdfile:
                    zero_frame = zip_reader.read(initfile)[0]
                    traj = dcd_reader.read(dcdfile, zero_frame)
                    traj.load_arrays()
            self.traj = traj
            return traj

    def get_log(self):
        log_file = self.fileroot +'/energy.log'
        self.log_table = pd.read_csv(log_file, sep='\t',header=0,skiprows=(lambda x: (x-1)%10000==0))
        return self.log_table

'''
Runs fundamental analyses on loaded simulation trajectory
This structure allows me to rapidly analyze individual simulations in parallel,
and then analyze them in aggregate.
'''
class Trajectory(SignacFile):
    # defaults to last frame
    def __init__(self,sp,index=None):
        SignacFile.__init__(self, sp, index)
        self.n = sp['n_gons']
        self.N = 10000
        self.frame = -1
        self.index_name = index
        self.recenter = False
        self.build_shapes()

    def build_shapes(self):
        n = self.n
        SC_ratio = 9
        radius = 1

        if self.n==360:
            self.a = 0
            self.r_circum = radius
            self.r_in = radius
            self.area = np.pi*radius**2
            self.sigma = 2*radius

        else:
            ## SHAPE DIMENSIONS
            self.a = (SC_ratio*2*np.pi*radius)/n
            # source : http://mathworld.wolfram.com/RegularPolygon.html
            self.r_circum = 0.5*self.a/np.sin(np.pi/n)
            self.r_in = 0.5*self.a/np.tan(np.pi/n)

            ## SHAPE VERTICES AND FORCE DIRECTION
            # Offsets shape vertices (s) and force (f) orientation so that force is perpendicular to faces when offset = 0
            # Force points in direction of vertice when offset = 1
            theta_2 = (np.pi)/n
            theta_s = theta_2
            vertices = []
            z = 0
            for i in range(n):
                x_s = self.r_circum*np.cos(theta_s)
                y_s = self.r_circum*np.sin(theta_s)
                vertices.append((x_s,y_s))
                theta_s += ((2*np.pi)/n)

            self.area = hoomd.dem.utils.area(vertices)
            equiarea_radius = np.sqrt(self.area/np.pi)
            self.sigma = 2*(equiarea_radius+radius)
        return

    def get_tau_per_frame(self):
        self.get_traj()
        self.init_frames = (self.sp['steps_init']+self.sp['steps_comp'])/self.sp['period_dump']
        self.tau_per_frame = self.sp['period_dump']*self.sp['dt']/self.sp['tau']

    # call to load trajectory
    def load_frame(self,frame=-1):
        self.frame = int(frame)
        self.get_traj()

        self.L = self.traj[self.frame].box.Lx
        self.fbox = box.Box(Lx=self.L, Ly=self.L, is2D=True)

        self.get_xy_pos()
        self.pos = np.ascontiguousarray(self.traj[self.frame].positions[:,:])
        self.orientations = np.ascontiguousarray(self.traj[self.frame].positions[:,2])

    # Note: does NOT recenter list of combined positions
    def get_xy_pos(self):
        if self.recenter:
            new_positions = Recenter(self.sp,index=self.index_name)
            new_positions.frame = self.frame
            new_positions.traj = self.traj
            new_positions.xy_recenter()
            self.x_pos = new_positions.x_pos
            self.y_pos = new_positions.y_pos
        else:
            self.x_pos = np.copy(self.traj[self.frame].positions[:,0])
            self.y_pos = np.copy(self.traj[self.frame].positions[:,1])
        return

    def build_grid(self,grids=50,frame=-1):
        x_pos = self.x_pos
        y_pos = self.y_pos
        x_grid = np.zeros(x_pos.shape)
        y_grid = np.zeros(y_pos.shape)
        grid = np.linspace(-self.L/2,self.L/2,grids)
        for x in range(grids-1):
            for y in range(grids-1):
                np.place(x_grid,(x_pos>grid[x])*(x_pos<=grid[x+1]),x)
                np.place(y_grid,(y_pos>grid[y])*(y_pos<=grid[y+1]),grids-y-2)
        self.y_grid = y_grid
        self.x_grid = x_grid

    def get_nn(self):
        radius = hoomd.dem.utils.rmax(self.doc['statepoint']['vertices'], radius=1.0, factor=1.0)
        rmax = 2.5*radius
        n_neigh = 8

        nn = freud.locality.NearestNeighbors(rmax=rmax,n_neigh=n_neigh,strict_cut=True)
        nn.compute(self.fbox, self.pos, self.pos)
        n_list = np.copy(nn.getNeighborList())
        num_particles = nn.getNRef()

        int_arr = np.ones(shape=n_list.shape, dtype=np.int32)
        int_arr[n_list > (num_particles-1)] = 0

        # sum along particle index axis to determine the number of neighbors per particle
        n_neighbors = np.sum(int_arr, axis=1)
        self.n_neighbors = n_neighbors
        return n_neighbors

    def get_localdensity(self,basis="sample"):
        ''' length of the local density
        '''
        samples = 100000

        if self.n==360:
            radius = self.doc['statepoint']['radius']*2*2**(1/6)
            area = self.area
        else:
            radius = hoomd.dem.utils.rmax(self.doc['statepoint']['vertices'], radius=1.0, factor=1.0)
            area = hoomd.dem.utils.spheroArea(self.doc['statepoint']['vertices'], radius=1.0)

        dens = None
        dens = density.LocalDensity(2.5*radius, area, 2*radius)
        if basis=="sample":
            sample = np.ascontiguousarray([[(np.random.rand()-.5)*self.L,(np.random.rand()-.5)*self.L,0] for n in range(samples)])
            dens.compute(self.fbox, sample, self.pos)
            self.local_density = np.copy(dens.getDensity())
        elif basis=="pos":
            dens.compute(self.fbox, self.pos, self.pos)
            self.local_density = np.copy(dens.getDensity())
        return self.local_density

    def get_localdensity_grid(self,grids=50):
        grid = np.linspace(-self.L/2,self.L/2,grids)
        x_pos = self.x_pos
        y_pos = self.y_pos

        self.get_localdensity(basis="pos")
        num_ld = self.local_density
        avg_ld = np.zeros((grids-1,grids-1))
        for x in range(grids-1):
            for y in range(grids-1):
                in_grid = (x_pos>grid[x])*(x_pos<=grid[x+1])*(y_pos>grid[y])*(y_pos<=grid[y+1])
                # note that this stores the y values in the wrong order, so need to flip when plotting
                if in_grid==[]: pass
                elif len(num_ld[in_grid])==1:
                    avg_ld[y,x]=num_ld[in_grid]
                else:
                    avg_ld[y,x] = np.mean(num_ld[in_grid])
        avg_ld = np.flip(avg_ld,0)
        self.local_density_array = np.copy(avg_ld)

    def plot_localdensity_grid(self,alpha=0.45):
        cMap, cNorm = util.get_colormap('ld')
        plt.figure(figsize=(10,8))
        plt.axis('off')
        plt.imshow(self.local_density_array,cmap=cMap,norm=cNorm,interpolation='none',alpha=alpha)

    def get_angular_displacement(self,start=-25,end=-1,grids=50):
        theta = self.traj[end].positions[:,2] - self.traj[start].positions[:,2]
        theta_diff = np.remainder(theta,(2*np.pi/self.n))
        cMap = cm.get_cmap('jet')
        cNorm = colors.Normalize(vmin=min(theta_diff),vmax=max(theta_diff))
        print(np.mean(theta_diff),min(theta_diff),max(theta_diff))

        plt.figure(figsize=(10,8))
        plt.axis('off')
        plt.scatter(self.x_pos,self.y_pos,color=cMap(cNorm(theta_diff)))

    def get_displacement(self,start=-50,end=-1,grids=50):
        if self.recenter:
            new_positions = Recenter(self.sp,index=self.index_name)
            new_positions.traj = self.traj
            new_positions.xy_recenter(frame=start,center_frame=end)
            x_start, y_start = new_positions.x_pos, new_positions.y_pos
            new_positions.xy_recenter(frame=end,center_frame=end)
            x_end, y_end = new_positions.x_pos, new_positions.y_pos
            x_diff = x_end-x_start
            y_diff = y_end-y_start
        else:
            x_diff = self.traj[end].positions[:,0]-self.traj[start].positions[:,0]
            y_diff = self.traj[end].positions[:,1]-self.traj[start].positions[:,1]
        x_diff, y_diff = rebox_distances(x_diff,y_diff,self.L)

        self.build_grid()

        loc = np.vstack((self.x_grid,self.y_grid,x_diff,y_diff)).T
        df = pd.DataFrame(loc,columns=['x','y','x_diff','y_diff'])

        occupied = pd.DataFrame(np.vstack((self.x_grid,self.y_grid)).T).drop_duplicates()
        grid_val = []
        for row in occupied.values:
            grid_val.append(df[(df['x']==row[0]) & (df['y']==row[1])].mean().values)
        grid_val = np.asarray(grid_val).reshape((-1,4))
        plt.quiver(grid_val[:,0],grid_val[:,1],grid_val[:,2],grid_val[:,3],linewidth=5)

    def get_distancetravelled(self,start=-50,end=-1):
        x_0 = self.traj[start].positions[:,0]
        x_1 = self.traj[end].positions[:,0]
        y_0 = self.traj[start].positions[:,1]
        y_1 = self.traj[end].positions[:,1]
        x_diff, y_diff = rebox_distances(x_1-x_0,y_1-y_0,self.L)
        self.distance_travelled = np.sqrt((x_diff)**2+(y_diff)**2)
        return self.distance_travelled

    def save_plots(self,directory,file_name):
        util.check_dir(directory)
        plt.savefig(directory+file_name, bbox_inches='tight')

    def get_cluster_sizes(self):
        cluster_traj = Recenter(self.sp,index=self.index_name)
        cluster_traj.frame = self.frame
        cluster_traj.traj = self.traj
        cluster_traj.cluster_analysis()
        trajectory.cluster_sizes = cluster_traj.clusterProps.getClusterSizes()

'''
Recenters simulatino on the space -L/2 to L/2
'''
class Recenter(Trajectory):
    def __init__(self,sp,index=None,traj=None):
        Trajectory.__init__(self, sp, index)

    def cluster_analysis(self):
        try: radius = hoomd.dem.utils.rmax(self.doc['statepoint']['vertices'], radius=1.0, factor=1.0)
        except: radius = self.doc['statepoint']['radius']*1.5
        rcut = 2.5*radius

        # Calls cluster class & compute the clusters for the given set of points
        self.clusterClass = freud.cluster.Cluster(self.fbox, rcut)
        self.clusterClass.computeClusters(self.pos)
        clusteridx = self.clusterClass.getClusterIdx()

        # Calculates properties of clusters (calls class first)
        self.clusterProps = freud.cluster.ClusterProperties(self.fbox)
        self.clusterProps.computeProperties(self.pos,clusteridx)

    def get_largestCOM(self,frame=None):
        self.pos = np.ascontiguousarray(self.traj[frame].positions[:,:])
        # clusterProps = self.cluster_analysis() # pretty sure this isn't used, but if it breaks, this was where
        cluster_sizes = self.clusterProps.getClusterSizes()
        cluster_COM = self.clusterProps.getClusterCOM()

        largestclusteridx = np.argmax(cluster_sizes)
        a = cluster_COM[largestclusteridx]
        x,y = a[0],a[1]
        self.center_largestCOM = (x,y)

    def xy_recenter(self,traj=None,frame=None,center_frame=None):
        if traj is not None: self.traj = traj
        if center_frame is None: center_frame = self.frame
        if frame is None: frame = self.frame

        self.L = self.traj[self.frame].box.Lx
        self.fbox = box.Box(Lx=self.L, Ly=self.L, is2D=True)
        self.get_largestCOM(frame=center_frame)
        self.x_pos, self.y_pos = rebox_positions(self.traj[frame].positions[:,0],self.traj[frame].positions[:,1],self.L,self.center_largestCOM)
