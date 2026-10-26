# import numpy as np
# from scipy.fft import dct
# import scipy.interpolate as intp
# from numpy.typing import NDArray

# def calc_gij(self, aij: NDArray[np.int64]) -> NDArray[np.complex128]:
#     """
#     Calculates the scattering factor product g_ij(Q) with Q-dependence.

#     This function calculates the scattering factor product g_ij(Q), which accounts for
#     the Q-dependence of the atomic scattering factors.  This is important for accurate
#     PDF calculations, especially when dealing with materials containing elements with
#     significant differences in their scattering factors.  The calculation involves
#     obtaining the atomic scattering factors f0, f1, and f2, and then performing
#     a discrete cosine transform (DCT) to obtain the real-space representation.

#     Parameters
#     ----------
#     self : Simulator
#         The Simulator object, which must contain the atomic scattering factors
#         (f0s, f1f2s), atomic symbols, and relevant parameters (qmax, number_scf_function, pad).
#     aij : NDArray[np.int64]
#         A 2D array representing the atomic pairs.  Each row contains the indices
#         of the two atoms in a pair.

#     Returns
#     -------
#     NDArray[np.complex128]
#         A 2D NumPy array representing g_ij(r), the scattering factor product
#         in real space. The shape is (n_pairs, n_r_points).

#     Notes
#     -----
#     The function performs the following steps:
#     1.  Creates a Q-space vector.
#     2.  Retrieves atomic scattering factors f0, f1, f2 for each unique atom.
#     3.  Calculates the scattering factor product for each atomic pair.
#     4.  Calculates the average scattering factor.
#     5.  Normalizes the scattering factor product.
#     6.  Applies a Discrete Cosine Transform (DCT) to obtain the real-space
#         representation.
#     7.  Interpolates the result to match the desired r-space grid.
#     """
#     q = np.linspace(0, self["qmax"], self["number_scf_function"])
#     symbols_all = self.phase.model["symbol"]
#     sfs = np.zeros((symbols_all.size, q.size)) + 0.0j
#     fij = np.ones(np.append(aij.shape, q.size)).astype(float) + 0.0j

#     symbols = np.unique(symbols_all)
#     for atom in symbols:
#         f0 = self.f0s[atom](q)
#         f1, f2 = self.f1f2s[atom](self["energy"])
#         sf = f0 + f1 + 1.0j * f2
#         sfs[np.where(symbols_all == atom)] = sf
#         fij[np.where(aij == atom)] = sf

#     gij = fij[:, 0] * fij[:, 1]
#     occ = self.phase.model["occ"]
#     ave = np.average(sfs * occ[:, np.newaxis], axis=0)
#     gij /= np.square(ave)  # fij/<cf>2
#     gij = np.abs(gij)

#     number_scf_function = self["number_scf_function"] * self["pad"]
#     gij = dct(gij, n=number_scf_function, type=2, norm=None, axis=1) / (
#         2.0 * number_scf_function
#     )

#     if (self.r[1] - self.r[1]) != (self["qmax"] * np.float64(self["pad"])):
#         r = (
#             np.arange(0, np.float64(number_scf_function), 1.0)
#             * np.pi
#             / (self["qmax"] * np.float64(self["pad"]))
#         )
#         r_new = np.arange(r[0], r[-1], self.r[1] - self.r[0])
#         gij_new = np.empty((0, r_new.size))

#         for g in gij:
#             gn = intp.interp1d(r, g, fill_value=0.0)(r_new)
#             gij_new = np.append(gij_new, gn.reshape(1, -1), axis=0)

#         norm = np.sum(gij, axis=-1)
#         gij = gij_new
#         norm2 = np.sum(gij, axis=-1)
#         gij = gij / norm2[:, np.newaxis] * norm[:, np.newaxis]

#     gij = np.append(gij[:, :0:-1], gij, axis=-1)
#     return gij



# def calc_scattering_factors(self) -> NDArray[np.complex128]:
#     """
#     Calculates the atomic scattering factors with Q-dependence.

#     This function calculates the atomic scattering factors f(Q) for all unique atomic
#     symbols in the Phase object.  It combines the contributions from the
#     normal scattering factor (f0) and the anomalous scattering factors (f1, f2),
#     which are energy-dependent.

#     Parameters
#     ----------
#     self : Simulator
#         The Simulator object, which must contain the atomic symbols,
#         the Q-values, and the functions to calculate f0, f1, and f2
#         (self.f0s, self.f1f2s).

#     Returns
#     -------
#     numpy.ndarray
#         A 2D NumPy array of the atomic scattering factors f(Q) for each atom.
#         The shape is (n_atoms, n_q_points). The values are complex.
#     """
#     symbols = np.unique(self.symbols)
#     q = np.linspace(0, self["qmax"], self["number_scf_function"])
#     sfs = np.zeros((self.symbols.size, q.size)) + 0.0j

#     for atom in symbols:
#         f0 = self.f0s[atom](q)
#         f1, f2 = self.f1f2s[atom](self["energy"])
#         sf = f0 + f1 + 1.0j * f2
#         sfs[np.where(self.symbols == atom)] = sf

#     return sfs
