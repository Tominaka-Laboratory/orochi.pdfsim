import numpy as np
from numba import njit, prange, f8


@njit(
    f8[:](
        f8[:], f8[:], f8[:], f8[:], f8[:]
    ),
    fastmath=True,
    parallel=False,
    cache=True,
)
def gr_for_one_pair_type(
    r, rij, cij, sigma_ij, sigma_r_dependence
):
    """
    Calculates the pair distribution function as a histogram.

    This function is implemented using Numba and float64 precision for performance.
    It sums the contributions of individual Gaussian-like peaks to a global histogram.
    

    Parameters
    ----------
    r : ndarray, float64, shape (n_r,)
        The radial distances (x-axis bins) at which to evaluate the pair distribution function.
        Assumed to be uniformly spaced.
    rij : ndarray, float64, shape (n_pairs,)
        The distances between atom pairs (i, j), serving as the center of each Gaussian peak.
    cij : ndarray, float64, shape (n_pairs,)
        The number concentrations of atom pairs (i, j), scaling the height of each peak.
    sigma_ij : ndarray, float64, shape (n_pairs,)
        The intrinsic atomic displacement parameters (ADPs) or mean-square displacements of atom pairs (i, j).
        This contributes to the broadening of each peak.
    sigma_r_dependence : ndarray, float64, shape (n_r,)
        A scaling factor for `sigma_ij` at different radial distances, interpolated based on `rij`.
        This array defines the additional broadening of the peaks across `r`.

    Returns
    -------
    gr : ndarray, float64, shape (n_r,)
        The calculated pair distribution function, g(r), as a histogram.
    """
    rstep = r[1] - r[0]
    rmax_id = r.size - 1

    gr = np.zeros(r.size, dtype=r.dtype)
    pair_num = rij.size

    interp_sigma_r_dependence = np.interp(rij, r, sigma_r_dependence)

    for m in prange(pair_num):
        if rij[m] == 0:
            continue

        sigma_gauss = sigma_ij[m] * interp_sigma_r_dependence[m]

        center_bin_idx = (rij[m] - r[0]) // rstep + 1
        peak_data_point = np.ceil(5. * sigma_gauss / rstep)
        if peak_data_point < 5:
            peak_data_point = 5

        min_idx = int(center_bin_idx - peak_data_point)
        max_idx = int(center_bin_idx + peak_data_point)

        if min_idx > rmax_id or max_idx < 0:
            continue
        elif min_idx < 0:
            min_idx = 0
        elif max_idx > rmax_id:
            max_idx = rmax_id
        
        if sigma_gauss == 0:
            continue

        r_values_chunk = r[min_idx : max_idx]
        exponent_values = -np.square((r_values_chunk - rij[m]) / sigma_gauss) / 2.0
        convolved_shape_chunk = np.exp(exponent_values)
    
        convolved_shape_sum = np.sum(convolved_shape_chunk)
        if convolved_shape_sum == 0: 
            continue
        gr[min_idx : max_idx] += convolved_shape_chunk / convolved_shape_sum * cij[m]

    return gr

