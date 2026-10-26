from numba import (
    jit,
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


@njit(
    f4[:](
        f4[:], f4[:], f4[:], f4[:], f4[:, :], i4[:], i4[:], i4[:], i4[:], f4[:]
    ),
    fastmath=True,
    parallel=True,
    cache=True,
)
def calc_histogram(
    r, rij, cij, uij, gij, same_aij_groups, same_uij_groups, bmin, bmax, r_rij
):
    """
    Calculates the pair distribution function as a histogram.

    This function is implemented using Numba and float32 precision for performance.

    Parameters
    ----------
    r : ndarray, float32, shape (n_r,)
        The radial distances at which to evaluate the pair distribution function.
    rij : ndarray, float32, shape (n_pairs,)
        The distances between atom pairs (i, j).
    cij : ndarray, float32, shape (n_pairs,)
        The number concentrations of atom pairs (i, j).
    uij : ndarray, float32, shape (n_pairs,)
        The atomic displacement parameters (ADPs) or mean-square displacements of atom pairs (i, j).
    gij : ndarray, float32, shape (n_pairs, n_r_peak)  # n_r_peak: Number of r points for a single peak
        The Fourier transform of the product of scattering factors for atom pairs (i, j).
        Here, Q-dispersion of scattering factors is considered.
        Each element `gij[k, l]` represents the Fourier transform of f_i * f_j for pair `k`
        evaluated at the l-th r value used for the single-peak simulation.
        This corresponds to one peak of the pair distribution function for a single i-j pair.
        `n_r_peak` determines the range of radial distances used to define the shape of this peak.
    same_aij_groups : ndarray, int32, shape (n_groups_a + 1,)
        Indices that define the boundaries of groups of atom pairs with the same element combination
        (e.g., all Si-Si pairs). The values represent the starting index (inclusive) of each group
        in the `rij`, `cij`, `uij`, and `gij` arrays. The last element is the total number of pairs.
    same_uij_groups : ndarray, int32, shape (n_groups_u + 1,)
        Indices that define the boundaries of groups of atom pairs with the same ADPs.
        Similar to `same_aij_groups`, the values represent the starting index of each group.
        The last element is the total number of pairs.
    bmin : ndarray, int32, shape (n_r,)
        The minimum bin index for each r value.
    bmax : ndarray, int32, shape (n_r,)
        The maximum bin index for each r value.
    r_rij : ndarray, float32, shape (n_pairs,)
        The radial distance associated with each atom pair, used for peak broadening calculations.

    Returns
    -------
    gr : ndarray, float32, shape (n_r,)
        The calculated pair distribution function, g(r).
    """
    same_uij_number = len(same_uij_groups)
    same_uij_start = np.zeros(same_uij_number, dtype=np.uint32)
    same_uij_end = np.zeros(same_uij_number, dtype=np.uint32)

    for idx in range(same_uij_number):
        if idx != 0:
            same_uij_start[idx] = same_uij_end[idx - 1]
        same_uij_end[idx] = same_uij_groups[idx] + same_uij_start[idx]

    same_aij_number = len(same_aij_groups)
    gr_sum = np.zeros((same_aij_number, r.size), dtype=r.dtype)
    w = (gij.shape[1] - 1) // 2
    start_idx = 0

    for n in range(same_aij_number):
        end_idx = (np.where(same_uij_end == same_aij_groups[n]))[0][0] + 1
        gr_same_aij = np.zeros((end_idx - start_idx, r.size), dtype=r.dtype)

        for idx in prange(start_idx, end_idx):
            start = same_uij_start[idx]
            stop = same_uij_end[idx]

            Rij = rij[start:stop]
            Sij = uij[start:stop]
            Cij = cij[start:stop]
            Bij1 = bmin[start:stop]
            Bij2 = bmax[start:stop]

            Tij = np.exp(-np.square(r_rij / Sij[0]) / 2.0)
            Tij /= np.sum(Tij)

            M = Rij.size
            for m in range(M):
                if Rij[m] == 0:
                    continue
                gr_same_aij[idx - start_idx][Bij1[m] : Bij2[m]] += np.interp(
                    r[Bij1[m] : Bij2[m]], r_rij + Rij[m], Tij * Cij[m]
                )

        gr_sum[n] = np.convolve(np.sum(gr_same_aij, axis=0) / r, gij[n])[w:-w]
        start_idx = end_idx

    return np.sum(gr_sum, axis=0)


def calc_pdf(
    r, rij, cij, uij, gij, same_aij_groups, same_uij_groups, bmin, bmax, r_rij
):
    """
    Pre-processing function for calculating the pair distribution function, g(r), using Numba.

    This function prepares the input data for the Numba-optimized calculation of g(r).
    It may include steps such as data validation, unit conversions, or other pre-calculations.
    The data will be cast to float32 and int32 before being passed to the Numba function.

    Parameters
    ----------
    r : ndarray, float64, shape (n_r,)  # Or appropriate input type
        The radial distances at which to evaluate g(r).
    rij : ndarray, float64, shape (n_pairs,) # Or appropriate input type
        The distances between atom pairs (i, j).
    cij : ndarray, float64, shape (n_pairs,) # Or appropriate input type
        The number concentrations of atom pairs (i, j).
    uij : ndarray, float64, shape (n_pairs,) # Or appropriate input type
        The atomic displacement parameters (ADPs) or mean-square displacements of atom pairs (i, j).
    gij : ndarray, float64, shape (n_pairs,) # Or appropriate input type
        The Fourier transform of the product of scattering factors for atom pairs (i, j).
        Here, Q-dispersion of scattering factors is considered.
        Each element `gij[k, l]` represents the Fourier transform of f_i * f_j for pair `k`
        evaluated at the l-th r value used for the single-peak simulation.
        This corresponds to one peak of the pair distribution function for a single i-j pair.
        `n_r_peak` determines the range of radial distances used to define the shape of this peak.
    same_aij_groups : ndarray, int64, shape (n_groups_a + 1,) # Or appropriate input type
        Indices defining the boundaries of groups of atom pairs with the same element combination. The last element is the total number of pairs.
    same_uij_groups : ndarray, int64, shape (n_groups_u + 1,) # Or appropriate input type
        Indices defining the boundaries of groups of atom pairs with the same ADPs. The last element is the total number of pairs.
    bmin : ndarray, int64, shape (n_r,) # Or appropriate input type
        The minimum bin index for each r value.
    bmax : ndarray, int64, shape (n_r,) # Or appropriate input type
        The maximum bin index for each r value.
    r_rij : ndarray, float64, shape (n_pairs,) # Or appropriate input type
        The radial distance associated with each atom pair, used for peak broadening calculations.

    Returns
    -------
    gr : ndarray, float64, shape (n_r,)
        The calculated pair distribution function, g(r).
    """
    r = r.astype(np.float32)
    rij = rij.astype(np.float32)
    cij = cij.astype(np.float32)
    uij = uij.astype(np.float32)
    gij = gij.astype(np.float32)
    same_aij_groups = same_aij_groups.astype(np.int32)
    same_uij_groups = same_uij_groups.astype(np.int32)
    bmin = bmin.astype(np.int32)
    bmax = bmax.astype(np.int32)
    r_rij = r_rij.astype(np.float32)

    # print("number of points for pdf calculation:", cij.size)
    t0 = time.time()
    gr = calc_histogram(
        r,
        rij,
        cij,
        uij,
        gij,
        same_aij_groups,
        same_uij_groups,
        bmin,
        bmax,
        r_rij,
    )
    # print("time for cpu calculation", time.time()-t0)

    return gr.astype(np.float64)
