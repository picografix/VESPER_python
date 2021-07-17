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
# a file to contain all the functions 

# Mrc Class
class mrc_obj:
    def __init__(self, path):
        mrc = mrcfile.open(path)
        data = mrc.data
        header = mrc.header
        self.xdim = int(header.nx)
        self.ydim = int(header.ny)
        self.zdim = int(header.nz)
        self.xwidth = mrc.voxel_size.x
        self.ywidth = mrc.voxel_size.y
        self.zwidth = mrc.voxel_size.z
        self.cent = [
            self.xdim * 0.5,
            self.ydim * 0.5,
            self.zdim * 0.5,
        ]
        self.orig = {"x": header.origin.x, "y": header.origin.y, "z": header.origin.z}
        self.data = np.swapaxes(copy.deepcopy(data), 0, 2)
        self.dens = data.flatten()
        self.vec = np.zeros((self.xdim, self.ydim, self.zdim, 3), dtype="float32")
        self.dsum = None
        self.Nact = None
        self.ave = None
        self.std_norm_ave = None
        self.std = None

# set Vox Size
def mrc_set_vox_size(mrc, th=0.01, voxel_size=7.0):

    # set shape and size
    size = mrc.xdim * mrc.ydim * mrc.zdim
    shape = (mrc.xdim, mrc.ydim, mrc.zdim)

    # if th < 0 add th to all value
    if th < 0:
        mrc.dens = mrc.dens - th
        th = 0.0

    # Trim all the values less than threshold
    mrc.dens[mrc.dens < th] = 0.0
    mrc.data[mrc.data < th] = 0.0

    # calculate dmax distance for non-zero entries
    non_zero_index_list = np.array(np.nonzero(mrc.data)).T
    #non_zero_index_list[:, [2, 0]] = non_zero_index_list[:, [0, 2]]
    cent_arr = np.array(mrc.cent)
    d2_list = np.linalg.norm(non_zero_index_list - cent_arr, axis=1)
    dmax = max(d2_list)

    # dmax = math.sqrt(mrc.cent[0] ** 2 + mrc.cent[1] ** 2 + mrc.cent[2] ** 2)

    print("#dmax=" + str(dmax / mrc.xwidth))
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
    mrc_set.xwidth = mrc_set.ywidth = mrc_set.zwidth = voxel_size
    mrc_set.dens = np.zeros((mrc_set.xdim ** 3, 1))
    mrc_set.vec = np.zeros((new_xdim, new_xdim, new_xdim, 3), dtype="float32")
    mrc_set.data = np.zeros((new_xdim, new_xdim, new_xdim))

    print(
        "Nvox= "
        + str(mrc_set.xdim)
        + ", "
        + str(mrc_set.ydim)
        + ", "
        + str(mrc_set.zdim)
    )
    print(
        "cent= " + str(new_cent[0]) + ", " + str(new_cent[1]) + ", " + str(new_cent[2])
    )
    print(
        "ori= "
        + str(new_orig["x"])
        + ", "
        + str(new_orig["y"])
        + ", "
        + str(new_orig["z"])
    )

    return mrc, mrc_set

# helper calc function

@numba.jit(nopython=True)
def calc(stp, endp, pos, mrc1_data, fsiv):
    dtotal = 0
    pos2 = [0.0] * 3

    for xp in range(stp[0], endp[0]):
        rx = float(xp) - pos[0]
        rx = rx ** 2
        for yp in range(stp[1], endp[1]):
            ry = float(yp) - pos[1]
            ry = ry ** 2
            for zp in range(stp[2], endp[2]):
                rz = float(zp) - pos[2]
                rz = rz ** 2
                d2 = rx + ry + rz
                v = mrc1_data[xp][yp][zp] * math.exp(-1.5 * d2 * fsiv)
                dtotal += v
                pos2[0] += v * float(xp)
                pos2[1] += v * float(yp)
                pos2[2] += v * float(zp)
    return dtotal, pos2

# fastVec

def fastVEC(mrc1, mrc2, dreso=16.0):

    xydim = mrc1.xdim * mrc1.ydim
    Ndata = mrc2.xdim * mrc2.ydim * mrc2.zdim

    print(len(mrc2.dens))

    print("#Start VEC")
    gstep = mrc1.xwidth
    fs = (dreso / gstep) * 0.5
    fs = fs ** 2
    fsiv = 1.0 / fs
    fmaxd = (dreso / gstep) * 2.0
    print("#maxd= {fmaxd}".format(fmaxd=fmaxd))
    print("#fsiv= " + str(fsiv))

    dsum = 0.0
    Nact = 0

    list_d = []

    for x in tqdm(range(mrc2.xdim),disable=True):
        for y in range(mrc2.ydim):
            for z in range(mrc2.zdim):
                stp = [0] * 3
                endp = [0] * 3
                ind2 = 0
                ind = 0

                pos = [0.0] * 3
                pos2 = [0.0] * 3
                ori = [0.0] * 3

                tmpcd = [0.0] * 3

                v, dtotal, rd = 0.0, 0.0, 0.0

                pos[0] = (
                    x * mrc2.xwidth + mrc2.orig["x"] - mrc1.orig["x"]
                ) / mrc1.xwidth
                pos[1] = (
                    y * mrc2.xwidth + mrc2.orig["y"] - mrc1.orig["y"]
                ) / mrc1.xwidth
                pos[2] = (
                    z * mrc2.xwidth + mrc2.orig["z"] - mrc1.orig["z"]
                ) / mrc1.xwidth

                ind = mrc2.xdim * mrc2.ydim * z + mrc2.xdim * y + x

                # check density

                if (
                    pos[0] < 0
                    or pos[1] < 0
                    or pos[2] < 0
                    or pos[0] >= mrc1.xdim
                    or pos[1] >= mrc1.ydim
                    or pos[2] >= mrc1.zdim
                ):
                    mrc2.dens[ind] = 0.0
                    mrc2.vec[x][y][z][0] = 0.0
                    mrc2.vec[x][y][z][1] = 0.0
                    mrc2.vec[x][y][z][2] = 0.0
                    continue

                if mrc1.data[int(pos[0])][int(pos[1])][int(pos[2])] == 0:
                    mrc2.dens[ind] = 0.0
                    mrc2.vec[x][y][z][0] = 0.0
                    mrc2.vec[x][y][z][1] = 0.0
                    mrc2.vec[x][y][z][2] = 0.0
                    continue

                ori[0] = pos[0]
                ori[1] = pos[1]
                ori[2] = pos[2]

                # Start Point
                stp[0] = int(pos[0] - fmaxd)
                stp[1] = int(pos[1] - fmaxd)
                stp[2] = int(pos[2] - fmaxd)

                # set start and end point
                if stp[0] < 0:
                    stp[0] = 0
                if stp[1] < 0:
                    stp[1] = 0
                if stp[2] < 0:
                    stp[2] = 0

                endp[0] = int(pos[0] + fmaxd + 1)
                endp[1] = int(pos[1] + fmaxd + 1)
                endp[2] = int(pos[2] + fmaxd + 1)

                if endp[0] >= mrc1.xdim:
                    endp[0] = mrc1.xdim
                if endp[1] >= mrc1.ydim:
                    endp[1] = mrc1.ydim
                if endp[2] >= mrc1.zdim:
                    endp[2] = mrc1.zdim

                # setup for numba acc
                stp_t = List()
                endp_t = List()
                pos_t = List()
                [stp_t.append(x) for x in stp]
                [endp_t.append(x) for x in endp]
                [pos_t.append(x) for x in pos]

                # compute the total density
                dtotal, pos2 = calc(stp_t, endp_t, pos_t, mrc1.data, fsiv)

                mrc2.dens[ind] = dtotal
                mrc2.data[x][y][z] = dtotal

                if dtotal == 0:
                    mrc2.vec[x][y][z][0] = 0.0
                    mrc2.vec[x][y][z][1] = 0.0
                    mrc2.vec[x][y][z][2] = 0.0
                    continue

                rd = 1.0 / dtotal

                pos2[0] *= rd
                pos2[1] *= rd
                pos2[2] *= rd

                tmpcd[0] = pos2[0] - pos[0]
                tmpcd[1] = pos2[1] - pos[1]
                tmpcd[2] = pos2[2] - pos[2]

                dvec = math.sqrt(tmpcd[0] ** 2 + tmpcd[1] ** 2 + tmpcd[2] ** 2)

                if dvec == 0:
                    dvec = 1.0

                rdvec = 1.0 / dvec

                mrc2.vec[x][y][z][0] = tmpcd[0] * rdvec
                mrc2.vec[x][y][z][1] = tmpcd[1] * rdvec
                mrc2.vec[x][y][z][2] = tmpcd[2] * rdvec

                dsum += dtotal
                Nact += 1

    print("#End LDP")
    print(dsum)
    print(Nact)

    mrc2.dsum = dsum
    mrc2.Nact = Nact
    mrc2.ave = dsum / float(Nact)
    mrc2.std = np.linalg.norm(mrc2.dens[mrc2.dens > 0])
    mrc2.std_norm_ave = np.linalg.norm(mrc2.dens[mrc2.dens > 0] - mrc2.ave)

    print(
        "#MAP AVE={ave} STD={std} STD_norm={std_norm}".format(
            ave=mrc2.ave, std=mrc2.std, std_norm=mrc2.std_norm_ave
        )
    )
    # return False
    return mrc2


# rotation and FFTW
def rot_pos(vec, angle, inv=False):
    r = R.from_euler("zyx", angle, degrees=True)
    if inv:
        r = r.inv()
    rotated_vec = r.apply(vec)
    return rotated_vec

@numba.jit(nopython=True)
def rot_pos_mtx(mtx, vec):
    mtx = mtx.astype(np.float64)
    vec = vec.astype(np.float64)
    return vec @ mtx

@numba.jit(nopython=True)
def rot_mrc_combine(old_pos, new_pos, new_vec_shape):
    # combined_arr = np.hstack((old_pos,new_pos))
    combined_arr = np.concatenate((old_pos, new_pos), axis=1)

    combined_arr = combined_arr[
        (combined_arr[:, 0] >= 0)
        & (combined_arr[:, 1] >= 0)
        & (combined_arr[:, 2] >= 0)
        & (combined_arr[:, 0] < new_vec_shape[0])
        & (combined_arr[:, 1] < new_vec_shape[0])
        & (combined_arr[:, 2] < new_vec_shape[0])
    ]

    return combined_arr

def rot_mrc(orig_mrc_data, orig_mrc_vec, new_vec_shape, angle):
    new_pos = np.array(
        np.meshgrid(
            np.arange(new_vec_shape[0]),
            np.arange(new_vec_shape[1]),
            np.arange(new_vec_shape[2]),
        )
    ).T.reshape(-1, 3)

    # set the center
    cent = int(new_vec_shape[0] / 2.0)

    # get relative positions from center
    new_pos = new_pos - cent
    #print(new_pos)

    # init the rotation by euler angle
    r = R.from_euler("ZYX", angle, degrees=True)
    mtx = r.as_matrix()
    
    #print(mtx)

    old_pos = rot_pos_mtx(np.flip(mtx.T), new_pos) + cent

    combined_arr = np.hstack((old_pos, new_pos))

    combined_arr = combined_arr[
        (combined_arr[:, 0] >= 0)
        & (combined_arr[:, 1] >= 0)
        & (combined_arr[:, 2] >= 0)
        & (combined_arr[:, 0] < new_vec_shape[0])
        & (combined_arr[:, 1] < new_vec_shape[0])
        & (combined_arr[:, 2] < new_vec_shape[0])
    ]

    combined_arr = combined_arr.astype(np.int32)

    #print(combined_arr)
    #print(combined_arr.shape)

    # combined_arr = rot_mrc_combine(old_pos, new_pos, new_vec_shape)

    index_arr = combined_arr[:, 0:3]

    # print(index_arr)

    # print(np.count_nonzero(orig_mrc_data))
    dens_mask = orig_mrc_data[index_arr[:, 0], index_arr[:, 1], index_arr[:, 2]] != 0.0

    # print(dens_mask.shape)
    # print(dens_mask)

    non_zero_rot_list = combined_arr[dens_mask]

    #print(non_zero_rot_list.shape)
    #     with np.printoptions(threshold=np.inf):
    #         print(non_zero_rot_list[:, 0:3])

    non_zero_dens = orig_mrc_vec[
        non_zero_rot_list[:, 0], non_zero_rot_list[:, 1], non_zero_rot_list[:, 2]
    ]

    # print(non_zero_dens)

    #non_zero_dens[:, [2, 0]] = non_zero_dens[:, [0, 2]]
    new_vec = rot_pos_mtx(np.flip(mtx), non_zero_dens[:, 0:3])

    # print(new_vec)

    # init new vec array
    new_vec_array = np.zeros((new_vec_shape[0], new_vec_shape[1], new_vec_shape[2], 3))

    # print(new)

    # fill in the new data
    for vec, ind in zip(new_vec, (non_zero_rot_list[:, 3:6] + cent).astype(int)):
        new_vec_array[ind[0]][ind[1]][ind[2]][0] = vec[0]
        new_vec_array[ind[0]][ind[1]][ind[2]][1] = vec[1]
        new_vec_array[ind[0]][ind[1]][ind[2]][2] = vec[2]

    return new_vec_array

def find_best_trans(x, y, z):

    xyz_arr = x + y + z
    best = np.amax(xyz_arr)
    trans = np.unravel_index(xyz_arr.argmax(), xyz_arr.shape)

    return best, trans


def search_map_fft(mrc_target, mrc_search, TopN=10, ang=30, is_eval_mode=False):

    if is_eval_mode:
        print("#For Evaluation Mode")
        print("#Please use the same coordinate system and map size for map1 and map2.")
        print("#Example:")
        print("#In Chimera command line: open map1 and map2 as #0 and #1, then type")
        print("#> open map1.mrc")
        print("#> open map2.mrc")
        print("#> vop #1 resample onGrid #0")
        print("#> volume #2 save new.mrc")
        print("#Chimera will generate the resampled map2.mrc as new.mrc")
        return

    x1 = copy.deepcopy(mrc_target.vec[:, :, :, 0])
    y1 = copy.deepcopy(mrc_target.vec[:, :, :, 1])
    z1 = copy.deepcopy(mrc_target.vec[:, :, :, 2])

    #     x1 = np.swapaxes(x1 , 0, 2)
    #     y1 = np.swapaxes(y1 , 0, 2)
    #     z1 = np.swapaxes(z1 , 0, 2)

    #     x2 = copy.deepcopy(mrc_search.vec[:, :, :, 0])

    d3 = mrc_target.xdim ** 3

    rd3 = 1.0 / d3

    # print(f'{rd3:.20f}')

    X1 = np.fft.rfftn(x1)
    X1 = np.conj(X1)
    Y1 = np.fft.rfftn(y1)
    Y1 = np.conj(Y1)
    Z1 = np.fft.rfftn(z1)
    Z1 = np.conj(Z1)
    #     X2 = np.fft.rfftn(x2)

    #     X12 = np.multiply(X1,X2)

    x_angle = []
    y_angle = []
    z_angle = []

    i = 0
    while i < 360:
        x_angle.append(i)
        y_angle.append(i)
        i += ang

    i = 0
    while i <= 180:
        z_angle.append(i)
        i += ang

    angle_comb = np.array(np.meshgrid(x_angle, y_angle, z_angle)).T.reshape(-1, 3)

    mrc_angle_dict = {}

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=os.cpu_count() + 4
    ) as executor:
        trans_vec = {
            executor.submit(
                rot_mrc,
                mrc_search.data,
                mrc_search.vec,
                (mrc_search.xdim, mrc_search.ydim, mrc_search.zdim),
                angle,
            ): angle
            for angle in angle_comb
        }
        for future in concurrent.futures.as_completed(trans_vec):
            angle = trans_vec[future]
            mrc_angle_dict[tuple(angle)] = future.result()

    #     for angle in tqdm(angle_comb, desc="Rotation"):
    #         rot_result = rot_mrc(
    #             mrc_search.data,
    #             mrc_search.vec,
    #             (mrc_search.xdim, mrc_search.ydim, mrc_search.zdim),
    #             angle,
    #         )
    #         mrc_angle_dict[tuple(angle)] = rot_result

    # fftw plans
    a = pyfftw.empty_aligned((x1.shape), dtype="float32")
    b = pyfftw.empty_aligned(
        (a.shape[0], a.shape[1], a.shape[2] // 2 + 1), dtype="complex64"
    )
    c = pyfftw.empty_aligned((x1.shape), dtype="float32")

    fft_object = pyfftw.FFTW(a, b, axes=(0, 1, 2))
    ifft_object = pyfftw.FFTW(
        b, c, direction="FFTW_BACKWARD", axes=(0, 1, 2), normalise_idft=False
    )

    angle_score = []

    XX = []

    for angle in tqdm(angle_comb, desc="FFT", disable=True):
        rot_mrc_vec = mrc_angle_dict[tuple(angle)]

        x2 = copy.deepcopy(rot_mrc_vec[..., 0])
        y2 = copy.deepcopy(rot_mrc_vec[..., 1])
        z2 = copy.deepcopy(rot_mrc_vec[..., 2])

        #         x2 = np.swapaxes(x2 , 0, 2)
        #         y2 = np.swapaxes(y2 , 0, 2)
        #         z2 = np.swapaxes(z2 , 0, 2)

        X2 = np.zeros_like(X1)
        np.copyto(a, x2)
        np.copyto(X2, fft_object(a))
        X12 = X1 * X2
        np.copyto(b, X12)
        x12 = np.zeros_like(x1)
        np.copyto(x12, ifft_object(b))

        Y2 = np.zeros_like(Y1)
        np.copyto(a, y2)
        np.copyto(Y2, fft_object(a))
        Y12 = Y1 * Y2
        np.copyto(b, Y12)
        y12 = np.zeros_like(y1)
        np.copyto(y12, ifft_object(b))

        Z2 = np.zeros_like(Z1)
        np.copyto(a, z2)
        np.copyto(Z2, fft_object(a))
        Z12 = Z1 * Z2
        np.copyto(b, Z12)
        z12 = np.zeros_like(z1)
        np.copyto(z12, ifft_object(b))

        if tuple(angle) == (0, 0, 30):
            XX = [x12, y12, z12]

        #         X2 = np.fft.rfftn(x2)
        #         X12 = X1 * X2
        #         x12 = np.fft.irfftn(X12, norm="forward")

        # #         if (tuple(angle) == (0,0,30)):
        # #             XX = [X12,x12]

        #         Y2 = np.fft.rfftn(y2)
        #         Y12 = Y1 * Y2
        #         y12 = np.fft.irfftn(Y12, norm="forward")

        #         Z2 = np.fft.rfftn(z2)
        #         Z12 = Z1 * Z2
        #         z12 = np.fft.irfftn(Z12, norm="forward")

        best, trans = find_best_trans(x12, y12, z12)

        angle_score.append([tuple(angle), best * rd3, trans])

    #     num_jobs = math.ceil(360 / ang) * math.ceil(360 / ang) * (180 // ang + 1)

    return angle_score



# 

# 

# 

# 

# 

# 

# 