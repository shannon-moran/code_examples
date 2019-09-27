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

import utils.utils as util
import utils.initgrid as initgrid
import utils.viz as viz

class Parallelize(object):
    def __init__(self,grid,function):
        self.grid = grid
        self.function = function
        self.input_partitions()

    def independent(self,i):
        sps = initgrid.get_sps(self.grid[i])
        num_process = len(sps)
        p = multiprocessing.Pool(processes = num_process)
        p.map_async(self.function,sps)
        p.close()
        p.join()
        return

    def dependent(self,i):
        sps = initgrid.get_sps(self.grid[i])
        num_process = len(sps)
        results = multiprocessing.Pool(processes = num_process).map_async(self.function,sps)
        self.results = results
        return

    def input_partitions(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            'section',
            type=str,
            help="What part of the statepoints to look at.")
        args = parser.parse_args()

        sec = str(args.section)
        grid = self.grid
        if sec=='all': i_range = np.arange(0,len(grid))
        if sec=='1': i_range = np.arange(0,len(grid)/3)
        if sec=='2': i_range = np.arange(len(grid)/3,2*len(grid)/3)
        if sec=='3': i_range = np.arange(2*len(grid)/3,len(grid))
        if sec=='EF1': i_range = np.array([0,2]) # EF, 5-8
        if sec=='VF1': i_range = np.array([1,3]) # VF, 5-8
        if sec=='EF2': i_range = np.array([4,6,8,10]) # EF, 5-8
        if sec=='VF2': i_range = np.array([5,7,9,11]) # VF, 5-8

        self.i_range = i_range
        self.sec = sec
        return
