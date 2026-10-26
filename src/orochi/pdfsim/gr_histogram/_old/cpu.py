#old one before refactoring

# import numpy as np


# def calc_gr_histogram(self, q_dependence=False):
#     """
#     Prepares for the pair distribution function g(r) histogram calculation using CPU.

#     This function prepares the histogram for calculating the pair distribution
#     function (PDF) g(r). It groups pair data to enhance computational
#     efficiency by performing similar calculations in batches. Based on the
#     `q_dependence` parameter, it selects the appropriate structure factor
#     and PDF calculation functions.

#     Parameters
#     ----------
#     self : Calculator
#         An instance of the Calculator class for PDF calculation.
#     q_dependence : bool, optional
#         A flag indicating whether to consider Q-dependent structure factors.
#         Defaults to False.

#     Returns
#     -------
#     r : numpy.ndarray
#         Array of radial distances r.
#     gr : numpy.ndarray
#         The calculated histogram of the pair distribution function g(r).

#     Notes
#     -----
#     This function calculates the histogram through the following steps:

#     1.  Determines the range of pair data to be used for calculation using
#         the `set_r_range` function.
#     2.  Groups the pair data using the `grouping` function.
#     3.  Selects the appropriate structure factor and PDF calculation functions
#         based on the `q_dependence` parameter.
#     4.  Calculates the histogram of g(r) using the selected functions.
#     5.  Normalizes the histogram if necessary.

#     """
#     # data_in_r_range = set_r_range(self)
#     ai = self.pairs["symbol"][:, 0]
#     aj = self.pairs["symbol"][:, 1]
#     rij = self.pairs["distance"]
#     sij = self.pairs["adp"]
#     cij = self.pairs["occ"]
#     r = self.r
#     r_pair = self.r_pair


#     grouped_pairs = grouping(self, rij, cij, sij, ai, aj, r)
#     rij = grouped_pairs["rij"]
#     cij = grouped_pairs["cij"]
#     sij = grouped_pairs["sij"]
#     bin_min = grouped_pairs["bin_min"]
#     bin_max = grouped_pairs["bin_max"]
#     same_aij_groups = grouped_pairs["same_aij_groups"]
#     same_uij_groups = grouped_pairs["same_uij_groups"]
#     unique_aij = grouped_pairs["unique_aij"]

#     if q_dependence:
#         from ..structure_factor.q_depend import calc_gij
#         from .histogram.pdf_q_depend_cpu import calc_pdf
#     else:
#         from ..structure_factor.no_q_depend import calc_gij
#         from .histogram.pdf_no_q_depend_cpu import calc_pdf

#     gij = calc_gij(self, unique_aij)
#     gr = calc_pdf(
#         r,
#         rij,
#         cij,
#         sij,
#         gij,
#         same_aij_groups,
#         same_uij_groups,
#         bin_min,
#         bin_max,
#         r_pair,
#     )

#     if not q_dependence:
#         gr /= r

#     gr *= (self["qmax"] * self["pad"]) / np.pi
#     gr /= self.phase.model["occ"].size
#     return r, gr


# def calc_simulation_range(self, r_pair):
#     """
#     Calculates the simulation range.

#     This function determines the range of data to be used for PDF calculation
#     based on the provided `r_pair`. The calculation range is determined by the
#     minimum and maximum values of `self.r` extended by the range of `r_pair`.

#     Parameters
#     ----------
#     self : Calculator
#         An instance of the Calculator class for PDF calculation.
#     r_pair : numpy.ndarray
#         Array of pair distances r_ij.

#     Returns
#     -------
#     numpy.ndarray
#         An array of indices representing the simulation range.
#     """
#     eval_min = self.r[0] - r_pair[-1] 
#     eval_max = self.r[-1] + r_pair[-1]

#     if eval_min <= 0:
#         eval_min = 0
#     simulation_range = np.where((self.rij > eval_min) & (self.rij < eval_max))
#     return simulation_range[0]


# def make_bins(self, r, rij):
#     """
#     Generates histogram bins.

#     This function generates histogram bins based on the provided arrays of
#     radial distances `r` and pair distances `rij`. It calculates which bins
#     each `rij` value contributes to and returns the minimum and maximum
#     bin indices for each pair.

#     Parameters
#     ----------
#     self : Calculator
#         An instance of the Calculator class for PDF calculation.
#     r : numpy.ndarray
#         Array of radial distances r.
#     rij : numpy.ndarray
#         Array of pair distances r_ij.

#     Returns
#     -------
#     bin_min : numpy.ndarray
#         Array of minimum bin indices for each rij value.
#     bin_max : numpy.ndarray
#         Array of maximum bin indices for each rij value.
#     """
#     rstep = r[1] - r[0]
#     shift = (rij - r[0]) // rstep + 1
#     rmax_id = r.size - 1

#     n = self.num_p_peak - 1
#     bin_min = shift - n
#     bin_max = shift + n

#     bin_min[bin_min > rmax_id] = -1
#     bin_min[bin_min < 0] = 0
#     bin_max[bin_max < 0] = 1
#     return bin_min.flatten(), bin_max.flatten()


# # def grouping(self, rij, cij, sij, ai, aj, r):
# #     """
# #     Groups pair data.

# #     This function groups pair data based on the atomic pair type (ai, aj) and
# #     the mean-square relative displacement (sij) to enhance the efficiency of
# #     PDF calculation. It organizes the data so that similar calculations can
# #     be performed together.

# #     Parameters
# #     ----------
# #     self : Calculator
# #         An instance of the Calculator class for PDF calculation.
# #     rij : numpy.ndarray
# #         Array of pair distances r_ij.
# #     cij : numpy.ndarray
# #         Array of coherence lengths c_ij.
# #     sij : numpy.ndarray
# #         Array of mean-square relative displacements u_ij.
# #     ai : numpy.ndarray
# #         Array of indices for atom i.
# #     aj : numpy.ndarray
# #         Array of indices for atom j.
# #     r : numpy.ndarray
# #         Array of radial distances r.

# #     Returns
# #     -------
# #     grouped_pairs : dict
# #         A dictionary containing the grouped pair data.
# #         Keys include 'rij', 'cij', 'sij', 'bin_min', 'bin_max',
# #         'same_aij_groups', 'same_uij_groups', and 'unique_aij'.
# #     """
# #     grouped_pairs = {}

# #     # order the aij array
# #     aij = self.pair["symbol"]
# #     aij = np.sort(aij, axis=-1)
# #     self.pair_name = aij[:, 0].astype(object) + aij[:, 1].astype(object)
# #     self.pair_order = np.argsort(self.pair_name)
# #     aij = aij[self.pair_order]

# #     unique_aij, index_aij, counts_aij = np.unique(
# #         aij, return_index=True, return_counts=True, axis=0
# #     )

# #     index_sort = np.argsort(index_aij)
# #     index_aij = index_aij[index_sort]
# #     counts_aij = counts_aij[index_sort]
# #     self.pair_group_bounds = index_aij + counts_aij



# #     rij = self.pairs["distance"][sort]
# #     cij = self.pairs["occ"][sort]
# #     sij = self.pairs["adp"][sort]


# #     # find unique aij for scattering_factors


# #     rij = np.split(rij, same_aij_groups[:-1])
# #     cij = np.split(cij, same_aij_groups[:-1])
# #     sij = np.split(sij, same_aij_groups[:-1])

#     # bin_min = np.split(bin_min, same_aij_groups[:-1])
#     # bin_max = np.split(bin_max, same_aij_groups[:-1])

#     # rearrange the par by sorting them on the basis of simga values (adps)
#     # number_of_unique_aij = len(rij)
#     # same_uij_groups = np.array([])

#     # for x in range(number_of_unique_aij):
#     #     order_in_local_sigma = np.argsort(sij[x])
#     #     sorted_local_sigma = sij[x][order_in_local_sigma]
#     #     unique_local_sigma, counts = np.unique(
#     #         sorted_local_sigma, return_counts=True, axis=0
#     #     )

#     #     rij[x] = rij[x][order_in_local_sigma]
#     #     cij[x] = cij[x][order_in_local_sigma]
#     #     sij[x] = sorted_local_sigma
#     #     # bin_min[x] = bin_min[x][order_in_local_sigma]
#     #     # bin_max[x] = bin_max[x][order_in_local_sigma]

#     #     same_uij_groups = np.concatenate([same_uij_groups, counts], axis=0)

#     # rij = np.concatenate(rij, axis=0)
#     # cij = np.concatenate(cij, axis=0)
#     # sij = np.concatenate(sij, axis=0)
#     # bin_min = np.concatenate(bin_min, axis=0)
#     # bin_max = np.concatenate(bin_max, axis=0)



#     # grouped_pairs["rij"] = rij
#     # grouped_pairs["cij"] = cij
#     # grouped_pairs["sij"] = sij
#     # grouped_pairs["bin_min"] = bin_min
#     # grouped_pairs["bin_max"] = bin_max
#     # grouped_pairs["same_aij_groups"] = same_aij_groups
#     # # grouped_pairs["same_uij_groups"] = same_uij_groups
#     # # grouped_pairs["unique_aij"] = unique_aij

#     # return grouped_pairs


# # def set_r_range(self):
# #     """
# #     Sets the range of radial distances r to be used for calculation.

# #     This function determines the range of radial distances r to be used for
# #     PDF calculation. The range is calculated based on the specified r range
# #     and the width of the Gaussian function. It also calls the
# #     `calc_simulation_range` function to limit the calculation range.

# #     Parameters
# #     ----------
# #     self : Calculator
# #         An instance of the Calculator class for PDF calculation.

# #     Returns
# #     -------
# #     data_in_r_range : dict
# #         A dictionary containing the range of radial distances r to be used
# #         for calculation, along with related pair data.
# #         Keys include 'ai', 'aj', 'rij', 'sij', 'cij', 'r', and 'r_pair'.
# #     """
# #     # calculate evaluation range (from r[0] to r[-1] + gaussian width)

# #     # data_in_r_range = {}

# #     # rstep = self.r[1] - self.r[0]
# #     # r_extension = self.num_p_peak / self["qmax"] * 2.0
# #     # r_pair = np.arange(0.0, self.num_p_peak / self["qmax"], rstep)
# #     # r_pair = np.append(-r_pair[:0:-1], r_pair)

# #     r = np.arange(self.r[0], self.r[-1] + rstep + r_extension, rstep)

# #     # limit simulation_range
# #     sr = calc_simulation_range(self, r_pair)

# #     data_in_r_range["ai"] = self.ai[sr]
# #     data_in_r_range["aj"] = self.aj[sr]
# #     data_in_r_range["rij"] = self.rij[sr]

# #     sij = self.sij * self.peak_width
# #     data_in_r_range["sij"] = sij[sr]
# #     data_in_r_range["cij"] = self.cij[sr]
# #     data_in_r_range["r"] = r
# #     data_in_r_range["r_pair"] = r_pair

# #     return data_in_r_range
