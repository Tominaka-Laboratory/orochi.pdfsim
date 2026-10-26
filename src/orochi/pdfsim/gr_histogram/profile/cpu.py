import numpy as np
from numba import njit, prange, f8


@njit(f8[:](f8[:], f8[:], f8), fastmath=True, cache=True)
def sinc_convolution(gr_input, r_bins, q_sinc_param):
    """
    Applies a sinc function convolution to the input gr_input array.

    This function simulates the broadening effect of a sinc function
    in real space, using a convolution approach with a pre-calculated sinc kernel.

    Parameters
    ----------
    gr_input : ndarray, float64, shape (n_r,)
        The input pair distribution function (e.g., from Gaussian peaks only).
    q_sinc_param : float64
        The 'q' parameter for the sinc function: sin(q*x)/(q*x).
    r_bins : ndarray, float64, shape (n_r,)
        The radial distances (x-axis bins) corresponding to gr_input.
        Assumed to be uniformly spaced.

    Returns
    -------
    gr_convolved : ndarray, float64, shape (n_r,)
        The convolved pair distribution function.
    """
    if q_sinc_param == 0.0:
        return gr_input.copy()

    gr_convolved = np.zeros_like(gr_input)
    r_step = r_bins[1] - r_bins[0]
    num_r_bins = r_bins.size

    sinc_half_width_r = np.pi / q_sinc_param
    
    kernel_r_range = sinc_half_width_r * 30
    kernel_half_bins = int(kernel_r_range / r_step) + 1 
    
    if kernel_half_bins < 5:
        kernel_half_bins = 5
    elif kernel_half_bins > num_r_bins // 2:
        kernel_half_bins = num_r_bins // 2

    kernel_x_coords = np.arange(-kernel_half_bins, kernel_half_bins + 1) * r_step

    sinc_kernel = np.zeros_like(kernel_x_coords)
    for idx, x_val in enumerate(kernel_x_coords):
        sinc_arg = q_sinc_param * x_val
        if sinc_arg == 0.0:
            sinc_kernel[idx] = 1.0
        else:
            sinc_kernel[idx] = np.sin(sinc_arg) / sinc_arg
            
    sinc_kernel_sum = np.sum(sinc_kernel)
    if sinc_kernel_sum != 0:
        sinc_kernel /= sinc_kernel_sum
    
    for i in prange(num_r_bins):
        for k_idx, k_val in enumerate(range(-kernel_half_bins, kernel_half_bins + 1)):
            j = i - k_val
            
            if j >= 0 and j < num_r_bins:
                gr_convolved[i] += gr_input[j] * sinc_kernel[k_idx]

    return gr_convolved

@njit(f8[:](f8[:], f8[:], f8[:]), fastmath=True, cache=True, parallel=True)
def custom_kernel_convolution(gr_input, r_bins, custom_kernel):
    """
    Applies a custom kernel convolution to the input gr_input array.

    This function performs a convolution using a pre-defined kernel array provided
    by the user. The kernel is assumed to be centered around its middle element.

    Parameters
    ----------
    gr_input : ndarray, float64, shape (n_r,)
        The input pair distribution function.
    r_bins : ndarray, float64, shape (n_r,)
        The radial distances (x-axis bins) corresponding to gr_input.
        Assumed to be uniformly spaced.
    custom_kernel : ndarray, float64
        The 1D array representing the custom convolution kernel.
        It's assumed to be properly normalized if normalization is desired.
        The kernel should be symmetric around its center for typical applications,
        and its length determines the convolution's effective range.

    Returns
    -------
    gr_convolved : ndarray, float64, shape (n_r,)
        The convolved pair distribution function.
    """
    gr_convolved = np.zeros_like(gr_input)
    num_r_bins = r_bins.size
    kernel_half_bins = custom_kernel.size // 2

    for i in prange(num_r_bins):
        for k_idx in range(custom_kernel.size):
            j = i - (k_idx - kernel_half_bins)
            
            if j >= 0 and j < num_r_bins:
                gr_convolved[i] += gr_input[j] * custom_kernel[k_idx]

    return gr_convolved



# def calc_histogram_for_one_pair_type(
#     r, peak_data_point, rij, cij, sigma_ij, sigma_r_dependence
# ):
#     """
#     Calculates the pair distribution function as a histogram.

#     This function is implemented using Numba and float32 precision for performance.
#     It sums the contributions of individual Gaussian-like peaks to a global histogram.

#     Parameters
#     ----------
#     r : ndarray, float32, shape (n_r,)
#         The radial distances (x-axis bins) at which to evaluate the pair distribution function.
#         Assumed to be uniformly spaced.
#     peak_data_point : int32
#         The half-width of the sampling window (in number of `r` points) for each peak.
#         This determines how many points around `rij[m]` are considered for the Gaussian peak calculation.
#     rij : ndarray, float32, shape (n_pairs,)
#         The distances between atom pairs (i, j), serving as the center of each Gaussian peak.
#     cij : ndarray, float32, shape (n_pairs,)
#         The number concentrations of atom pairs (i, j), scaling the height of each peak.
#     sigma_ij : ndarray, float32, shape (n_pairs,)
#         The intrinsic atomic displacement parameters (ADPs) or mean-square displacements of atom pairs (i, j).
#         This contributes to the broadening of each peak.
#     sigma_r_dependence : ndarray, float32, shape (n_r,)
#         A scaling factor for `sigma_ij` at different radial distances, interpolated based on `rij`.
#         This array defines the additional broadening of the peaks across `r`.

#     Returns
#     -------
#     gr : ndarray, float32, shape (n_r,)
#         The calculated pair distribution function, g(r), as a histogram.
#     """
#     rstep = r[1] - r[0]
#     rmax_id = r.size - 1

#     gr = np.zeros(r.size, dtype=r.dtype)
#     pair_num = rij.size

#     # Pre-calculate peak width interpolation for all rij values
#     interp_sigma_r_dependence = np.interp(rij, r, sigma_r_dependence)

#     for m in prange(pair_num):
#         # Skip if the pair distance is zero
#         if rij[m] == 0:
#             continue
                
#         # Calculate the center bin index for the current peak
#         center_bin_idx = (rij[m] - r[0]) // rstep + 1
        
#         # Determine the start and end indices for the peak's contribution
#         min_idx = int(center_bin_idx - peak_data_point)
#         max_idx = int(center_bin_idx + peak_data_point)

#         # Boundary checks and adjustments for the peak's range
#         if min_idx > rmax_id or max_idx < 0:
#             continue # Peak is entirely outside the r range
#         elif min_idx < 0:
#             min_idx = 0 # Clip lower bound
#         elif max_idx > rmax_id:
#             max_idx = rmax_id # Clip upper bound
        
#         # Calculate the effective sigma for the current peak
#         sigma = sigma_ij[m] * interp_sigma_r_dependence[m]
        
#         # Handle zero sigma to avoid division by zero or infinite values in exp
#         if sigma == 0:
#             continue
#         r_values_chunk = r[min_idx : max_idx]

#         Tij_shape = np.exp(-np.square((r_values_chunk - rij[m]) / sigma) / 2.0) 

#         Tij_sum = np.sum(Tij_shape)
#         if Tij_sum == 0: 
#             continue
#         Tij_shape /= Tij_sum

#         gr[min_idx : max_idx] += Tij_shape * cij[m]

#     return gr
# def calc_pdf(self
# ):
#     """
#     Pre-processing function for calculating the pair distribution function, g(r), using Numba.

#     This function prepares the input data for the Numba-optimized calculation of g(r).
#     It may include steps such as data validation, unit conversions, or other pre-calculations.
#     The data will be cast to float32 and int32 before being passed to the Numba function.

#     After the Numba calculation of g(r).

#     Parameters
#     ----------
#     r : ndarray, float64, shape (n_r,)  # Or appropriate input type
#         The radial distances at which to evaluate g(r).
#     rij : ndarray, float64, shape (n_pairs,) # Or appropriate input type
#         The distances between atom pairs (i, j).
#     cij : ndarray, float64, shape (n_pairs,) # Or appropriate input type
#         The number concentrations of atom pairs (i, j).
#     sigma_ij : ndarray, float64, shape (n_pairs,) # Or appropriate input type
#         The atomic displacement parameters (ADPs) or mean-square displacements of atom pairs (i, j).
#     gij : ndarray, float64, shape (n_pairs,) # Or appropriate input type
#         The Fourier transform of the product of scattering factors for atom pairs (i, j).
#         Here, Q-dispersion of scattering factors is ignored, and simply the electron numbers of elements are used.
#         Each element `gij[k]` represents the Fourier transform of f_i * f_j for pair `k`.
#         This corresponds to the peak of the pair distribution function for a single i-j pair.
#     same_aij_groups : ndarray, int64, shape (n_groups_a + 1,) # Or appropriate input type
#         Indices defining the boundaries of groups of atom pairs with the same element combination. The last element is the total number of pairs.
#     same_uij_groups : ndarray, int64, shape (n_groups_u + 1,) # Or appropriate input type
#         Indices defining the boundaries of groups of atom pairs with the same ADPs. The last element is the total number of pairs.
#     bmin : ndarray, int64, shape (n_r,) # Or appropriate input type
#         The minimum bin index for each r value.
#     bmax : ndarray, int64, shape (n_r,) # Or appropriate input type
#         The maximum bin index for each r value.
#     r_pair : ndarray, float64, shape (n_pairs,) # Or appropriate input type
#         The radial distance associated with each atom pair, used for peak broadening calculations.

#     Returns
#     -------
#     gr : ndarray, float64, shape (n_r,)
#         The calculated pair distribution function, g(r)/r.
#     """
#     split_pairs = {}
#     for key, value in self.pairs.items():
#         ordered_value = value[self.pair_order]
#         split_value = np.split(ordered_value, self.pair_group_bounds[:-1])
#         split_pairs[key] = split_value

#     partial_gr = {}
#     r = self.r.astype(np.float32)
#     r_pair = self.r_pair.astype(np.float32)
#     for n, rij in enumerate(split_pairs["distance"]):
#         rij = rij.astype(np.float32)
#         cij = split_pairs["occ"][n].astype(np.float32)
#         sigma_ij = split_pairs["adp"][n].astype(np.float32)

#         gr = calc_histogram_for_one_pair_type(
#             r, r_pair, rij, cij, sigma_ij
#         ).astype(np.float64)
#         gr *= self.gij[n]
#         partial_gr[self.pair_name[n]] = gr
#     self.partial_gr = partial_gr










#     gr = calc_histogram(
#         r,
#         rij,
#         cij,
#         sigma_ij,
#         gij,
#         same_aij_groups,
#         same_uij_groups,
#         bmin,
#         bmax,
#         r_pair,
#     )

#     return gr.astype(np.float64)
