from numba import (
    jit,
    cuda,
    prange,
    void,
    float64,
    int64,
    float32,
    int32,
    njit,
    f8,
    f4,
    i8,
    i4,
)
import numpy as np
import time
import math


@cuda.jit(
    "void(f4[:], f4[:,:], f4[:,:], f4[:], f4[:], f4[:], f4[:])",
    fastmath=True,
    cache=True,
)
def calc_histogram(map, xyz, xyz_tvs, adp, gr, r, r_peak):

    tx = cuda.threadIdx.x  # thred index, 0 to bwx
    ty = cuda.threadIdx.y

    bwx = cuda.blockDim.x
    bwy = cuda.blockDim.y

    bx = cuda.blockIdx.x  # block index, 0 to gwx
    by = cuda.blockIdx.y

    gwx = cuda.gridDim.x
    gwy = cuda.gridDim.y

    x = tx + bx * bwx
    y = ty + by * bwy

    x_size = xyz.shape[0]
    y_size = xyz_tvs.shape[0]

    if (x >= x_size) | (y >= y_size):
        return

    r_step = r[1] - r[0]
    r_start = r[0]
    rps = (r_peak.size - 1) // 2

    sx = bwx * gwx
    sy = bwy * gwy

    for i in range(x, x_size, sx):
        if i >= x_size:
            continue

        xyz_i = xyz[i]
        map_i = map[i]
        adp_i = adp[i]

        for j in range(y, y_size, sy):
            if j >= y_size:
                continue

            xyz_j = xyz_tvs[j]
            map_j = map[int(j % x_size)]
            adp_j = adp[int(j % x_size)]
            cij = map_i * map_j
            uij = math.sqrt(adp_i + adp_j)

            tij_sum = 0.0
            for rp in r_peak:
                tij = math.exp(-1 / 2 * (rp / uij) ** 2.0)
                tij_sum += tij

            rij = 0.0
            for n in range(3):
                rij += math.pow(xyz_j[n] - xyz_i[n], 2.0)
            rij = math.sqrt(rij)

            if rij == 0:
                continue

            rij_s = rij - r_start
            ideal_bin = rij_s / r_step  # float
            lower_bin = int(math.floor(ideal_bin))
            upper_bin = int(math.ceil(ideal_bin))
            upper_ratio = ideal_bin - math.floor(ideal_bin)
            lower_ratio = 1 - upper_ratio

            # upper bins
            bin = upper_bin
            ratio = upper_ratio
            bmin = bin - rps
            bmax = bin + rps + 1
            dmin = 0
            dmax = gr.size

            if bmin < 0:
                dmin = -bin
                bmin = 0
            if bmax > gr.size:
                dmax = gr.size - bmax
                bmax = gr.size

            m = 0
            for b in range(bmin, bmax):
                rp = r_peak[dmin:dmax][m]
                tij = math.exp(-1 / 2 * (rp / uij) ** 2.0)
                cuda.atomic.add(gr, b, cij * tij * ratio / tij_sum)
                m += 1

            # lower bins
            bin = lower_bin
            ratio = lower_ratio
            bmin = bin - rps
            bmax = bin + rps + 1
            dmin = 0
            dmax = gr.size

            if bmin < 0:
                dmin = -bin
                bmin = 0
            if bmax > gr.size:
                dmax = gr.size - bmax
                bmax = gr.size

            m = 0
            for b in range(bmin, bmax):
                rp = r_peak[dmin:dmax][m]
                tij = math.exp(-1 / 2 * (rp / uij) ** 2.0)
                cuda.atomic.add(gr, b, cij * tij * ratio / tij_sum)
                m += 1


def calc_pdf(xyz, tvs, occ, f, adp, r, r_peak):
    # data
    print("cuda availability:", cuda.is_available())
    xyz_tvs = (
        (xyz[np.newaxis, :] + tvs[:, np.newaxis])
        .reshape(-1, 3)
        .astype(np.float32)
    )
    xyz = xyz.astype(np.float32)

    map = occ * f
    map = map.astype(np.float32)
    adp = adp.astype(np.float32)
    r = r.astype(np.float32)
    r_peak = r_peak.astype(np.float32)
    gr = np.zeros_like(r).astype(np.float32)

    x_size = xyz.shape[0]
    y_size = xyz_tvs.shape[0]

    # grid structures
    tpbx = 1
    tpby = 1
    # tpbz = 1
    blockdim = (tpbx, tpby)  # 22,22 #number of threads: max 512

    bpgx = math.ceil(x_size / tpbx)
    bpgy = math.ceil(y_size / tpby)
    if bpgx > 65535:
        bpgx = 65535
    if bpgy > 65535:
        bpgy = 65535
    griddim = (bpgx, bpgy)

    print("number of points for pdf calculation:", map.size, tvs.shape[0])
    t0 = time.time()

    # host to device
    map_gpu = cuda.to_device(map)
    xyz_gpu = cuda.to_device(xyz)
    xyz_tvs_gpu = cuda.to_device(xyz_tvs)
    adp_gpu = cuda.to_device(adp)
    r_gpu = cuda.to_device(r)
    rp_gpu = cuda.to_device(r_peak)
    gr_gpu = cuda.to_device(gr)

    calc_histogram[griddim, blockdim](
        map_gpu, xyz_gpu, xyz_tvs_gpu, adp_gpu, gr_gpu, r_gpu, rp_gpu
    )
    cuda.synchronize()
    gr = gr_gpu.copy_to_host()
    print("time for gpu calculation", time.time() - t0)

    gr /= np.average(map) ** 2.0

    return gr.astype(np.float64)
