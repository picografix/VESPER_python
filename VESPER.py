import copy
import math

import mrcfile
import numpy as np
import command
from CMD import CMD
# from Bio.PDB import *
# import numba as nb

class mrc_obj:
    def __init__(self, path):
        mrc = mrcfile.open(path)
        data = mrc.data
        header = mrc.header
        self.xdim = header.nx
        self.ydim = header.ny
        self.zdim = header.nz
        self.xwidth = mrc.voxel_size.x
        self.ywidth = mrc.voxel_size.y
        self.zwidth = mrc.voxel_size.z
        self.cent = [
            self.xdim * 0.5,
            self.ydim * 0.5,
            self.zdim * 0.5,
        ]
        self.orig = {"x": header.origin.x, "y": header.origin.y, "z": header.origin.z}
        self.dens = np.swapaxes(data, 0, 2)
        self.vec = None
        self.dsum = None
        self.Nact = None
        self.ave = None
        self.std_norm_ave = None
        self.std = None


def mrc_set_vox_size(mrc, th=0.0, voxel_size=7.0):

    # set shape and size
    size = mrc.xdim * mrc.ydim * mrc.zdim
    shape = (mrc.xdim, mrc.ydim, mrc.zdim)

    # if th < 0 add th to all value
    dens = mrc.dens.flatten()
    if th < 0:
        dens = dens - th
        th = 0.0

    # Trim all the values less than threshold
    dens[dens < th] = 0

    # calculate dmax distance for non-zero entries
    non_zero_index_list = np.nonzero(dens)
    index_3d = np.unravel_index(non_zero_index_list, shape)
    index_arr = np.array([index_3d[0][0], index_3d[1][0], index_3d[2][0]]).T
    cent_arr = np.array(mrc.cent)
    d2_list = np.linalg.norm(index_arr - cent_arr, axis=1)
    dmax = max(d2_list)

    #dmax = math.sqrt(mrc.cent[0] ** 2 + mrc.cent[1] ** 2 + mrc.cent[2] ** 2)
    dmax = dmax * mrc.xwidth

    # set new center
    new_cent = [
        mrc.cent[0] * mrc.xwidth + mrc.orig["x"],
        mrc.cent[1] * mrc.xwidth + mrc.orig["y"],
        mrc.cent[2] * mrc.xwidth + mrc.orig["z"],
    ]

    tmp_size = 2 * dmax / voxel_size

    # find the minimum size of the map
    b = y = 2 ** math.ceil(math.log2(tmp_size))
    while 1:
        while y < tmp_size:
            y = y * 3
            continue
        if y < b:
            b = y
        if y % 2 != 0:
            break
        y = y / 2

    new_xdim = int(b)

    # set new origins
    new_orig = {
        "x": new_cent[0] - 0.5 * new_xdim * voxel_size,
        "y": new_cent[1] - 0.5 * new_xdim * voxel_size,
        "z": new_cent[2] - 0.5 * new_xdim * voxel_size,
    }

    # create new mrc object
    mrc_set = copy.deepcopy(mrc)
    mrc_set.orig = new_orig
    mrc_set.xdim = mrc_set.ydim = mrc_set.zdim = new_xdim
    mrc_set.cent = new_cent

    return mrc_set

def mrc_vec(mrc, mrc_N, dreso=16.00):
    
    # set up filter
    gstep = mrc.xwidth
    fs = (dreso / gstep) * 0.5
    fs = fs ** 2
    fsiv = 1.0 / fs
    fmaxd = (dreso / gstep) * 2.0
    dsum = 0
    Nact = 0
    
    
    
    
    # TODO
    return


mrc1 = mrc_obj("Data\emd_8097.mrc")
mrc2 = mrc_obj("Data\ChainA_simulated_resample.mrc")

mrc_N1 = mrc_set_vox_size(mrc1)
mrc_N2 = mrc_set_vox_size(mrc2)

if mrc_N1.xdim > mrc_N2.xdim:
    mrc_N2.xdim = mrc_N2.ydim = mrc_N2.zdim = mrc_N1.xdim
    mrc_N2.cent = [
        mrc_N2.cent[0] - 0.5 * 7 * mrc_N2.xdim,
        mrc_N2.cent[1] - 0.5 * 7 * mrc_N2.xdim,
        mrc_N2.cent[2] - 0.5 * 7 * mrc_N2.xdim,
    ]
else:
    mrc_N1.xdim = mrc_N1.ydim = mrc_N1.zdim = mrc_N2.xdim
    mrc_N1.cent = [
        mrc_N1.cent[0] - 0.5 * 7 * mrc_N1.xdim,
        mrc_N1.cent[1] - 0.5 * 7 * mrc_N1.xdim,
        mrc_N1.cent[2] - 0.5 * 7 * mrc_N1.xdim,
    ]

# create a CMD object
cmd = CMD()
# command.input(cmd)
command.input(cmd)
def fastVEC(mrc1,mrc2):
  i,j,k,ind=0,0,0,0
  cnt=0
  xydim=mrc1.xdim * mrc1.ydim
  Ndata = mrc2.xdim * mrc2.ydim * mrc2.xdim
  print("#Start VEC")
  dreso = cmd.dreso
  gstep = mrc1.widthx
  fs = (dreso/gstep) * 0.5
  fsiv = 1.000 / fs
  fmaxd = (dreso / gstep) * 2.0
  print("#maxd= {fmaxd}\n".format(fmaxd=fmaxd))
  dsum=0
  Nact=0

  for x in range(mrc2.xdim):
    rx,ry,rz,d2=0.0,0.0,0.0,0.0
    for y in range(mrc2.ydim):
      for z in range(mrc2.zdim):
        stp=[0]*3
        endp = [0]*3
        ind2=0
        ind=0

        pos=[0]*3
        pos2=[0]*3
        ori=[0]*3

        tmpcd = [0]*3

        v,dtotal,rd=0.0,0.0,0.0

        pos[0] = (x* mrc2.widhtx + mrc2.orgxyz[0] - mrc1.orgxyz[0]) / mrc1.widhtx
        pos[1] = (y* mrc2.widhtx + mrc2.orgxyz[1] - mrc1.orgxyz[1]) / mrc1.widhtx
        pos[2] = (z* mrc2.widhtx + mrc2.orgxyz[2] - mrc1.orgxyz[2]) / mrc1.widhtx

        ind = mrc2.xdim * mrc2.ydim*z + mrc2.xdim*y + x

        # check density

        if pos[0] < 0 or pos[1] <0 or pos[2] < 0 or pos[0] >= mrc1.xdim or pos[1] >= mrc1.ydim or pos[2] >= mrc1.zdim:
          mrc2.dens[ind] = 0
          mrc2.vec[ind][0] = mrc2.vec[ind][1] = mrc2.vec[ind][2]=0.00
          continue

        ind0 = mrc1.xdim * mrc1.ydim * pos[2] + mrc1.xdim * pos[1] + pos[0]

        if(mrc1.dens[ind0]==0):
          mrc2.dens[ind] = 0
          mrc2.vec[ind][0] = mrc2.vec[ind][1] = mrc2.vec[ind][2] = 0.00
        ori[0] = pos[0];
        ori[1] = pos[1];
        ori[2] = pos[2];
        #Start Point
        stp[0] = int(pos[0] - fmaxd);
        stp[1] = int(pos[1] - fmaxd);
        stp[2] = int(pos[2] - fmaxd);

        if (stp[0] < 0):
          stp[0] = 0
        if (stp[1] < 0):
          stp[1] = 0
        if (stp[2] < 0):
          stp[2] = 0


        endp[0] = int(pos[0] + fmaxd + 1)
        endp[1] = int(pos[1] + fmaxd + 1)
        endp[2] = int(pos[2] + fmaxd + 1)

        if (endp[0] >= mrc1.xdim):
           endp[0] = mrc1.xdim
        if (endp[1] >= mrc1.ydim):
           endp[1] = mrc1.ydim
        if (endp[2] >= mrc1.zdim):
           endp[2] = mrc1.zdim

        dtotal = 0
        pos2[0] = pos2[1]= pos2[2] = 0

        for xp in range(stp[0],endp[0]):
          rx = float(yp-pos[1])
          rx = rx*rx
          for yp in range(stp[1],endp[1]):
            ry = float(yp-pos[1])
            ry=ry*ry
            for zp in range(stp[2],endp[2]):
              rz = float(zp-pos[2])
              rz=rz**2
              d2=rx+ry+rz
              ind2=xydim*zp+mrc1.xdim*yp+zp
              v=mrc1.dens[ind2]*math.exp(-1.5*d2*fsiv)
              dtotal+=v;
              pos2[0]+=v*float(xp);
              pos2[1]+=v*float(yp);
              pos2[2]+=v*float(zp);
  print("#End LDP")

  mrc2.dsum=dsum
  mrc2.Nact=Nact
  mrc2.ave=dsum/float(Nact)
  dsum=0
  dsum2=0.0
  for i in range(mrc2.xdim*mrc2.ydim*mrc2.zdim):
    if(mrc2.dens[i]>0):
      dsum+=mrc2.dens[i]*mrc2.dens[i]
      dsum2+=(mrc2.dens[i]-mrc2.ave)*(mrc2.dens[i]-mrc2.ave)
  mrc2.std_norm_ave=math.sqrt(dsum2)
  mrc2.std=math.sqrt(dsum)
  print("#MAP AVE={ave} STD={std} STD_norm={std_norm}".format(ave=mrc2.ave,std=mrc2.std,std_norm=mrc2.std_norm_ave))
  return False
