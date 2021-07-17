# imports
import time
import concurrent.futures
import copy
import math
import multiprocessing
import os

import mrcfile
import numba
import numpy as np
import pyfftw
import scipy.fft
from numba.typed import List
from scipy.spatial.transform import Rotation as R
from tqdm.notebook import tqdm

# import utils

from utils import *
from CMD import CMD
import command

start = time.time()
# set threads
pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
print("Threads Initialized")

# code begins here

cmd  = CMD() #creates a CMD object for input 
command.input(cmd) #takes input if provided otherwise runs with default input

# initialize mrc_obj
mrc1 = mrc_obj(cmd.file1)
mrc2 = mrc_obj(cmd.file2)
print("mrc Initialized")
# set vox size

mrc1, mrc_N1 = mrc_set_vox_size(mrc1)
mrc2, mrc_N2 = mrc_set_vox_size(mrc2)

if mrc_N1.xdim > mrc_N2.xdim:
    mrc_N2.xdim = mrc_N2.ydim = mrc_N2.zdim = mrc_N1.xdim

    mrc_N2.orig["x"] = mrc_N2.cent[0] - 0.5 * 7 * mrc_N2.xdim
    mrc_N2.orig["y"] = mrc_N2.cent[1] - 0.5 * 7 * mrc_N2.xdim
    mrc_N2.orig["z"] = mrc_N2.cent[2] - 0.5 * 7 * mrc_N2.xdim

else:
    mrc_N1.xdim = mrc_N1.ydim = mrc_N1.zdim = mrc_N2.xdim

    mrc_N1.orig["x"] = mrc_N1.cent[0] - 0.5 * 7 * mrc_N1.xdim
    mrc_N1.orig["y"] = mrc_N1.cent[1] - 0.5 * 7 * mrc_N1.xdim
    mrc_N1.orig["z"] = mrc_N1.cent[2] - 0.5 * 7 * mrc_N1.xdim
print("set voc size complete")
# run fastVEC
print("FastVec Started")
mrc_N1 = fastVEC(mrc1, mrc_N1)
mrc_N2 = fastVEC(mrc2, mrc_N2)
print("FastVec Complete")
# rotate mrc
print("Mrc Rotation Started")
mrc = rot_mrc(mrc_N2.data, mrc_N2.vec, (48, 48, 48), [0, 0, 30])
print("Mrc Rotation Complete")
#run  search map fftw
print("Search Map Started")
score_list = search_map_fft(mrc_N1, mrc_N2, ang=30)
print("Search Map End")
print(sorted(score_list, key=lambda x: x[1], reverse=True)[:10])

# end

print("process complete")

end = time.time()
print(f"Runtime of the program is {end - start}")