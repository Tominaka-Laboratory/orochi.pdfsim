#old one before refactoring

# import numpy as np
# from numpy.typing import NDArray

# def calc_gij(self, aij: NDArray[np.int64]) -> NDArray[np.float64]:
#     """
#     Calculates the scattering factor product g_ij (Q-independent).

#     This function calculates the scattering factor product g_ij, assuming that
#     the atomic scattering factors are independent of the scattering vector Q.
#     This simplification is often used for simpler models or when Q-dependence
#     is not significant.  The calculation uses the atomic numbers (zi) to
#     approximate the scattering factors.

#     Parameters
#     ----------
#     self : Simulator
#         The Simulator object, which must contain the atomic numbers (zi)
#         and atomic symbols.
#     aij : NDArray[np.int64]
#         A 2D array representing the atomic pairs. Each row contains the indices
#         of the two atoms in a pair.

#     Returns
#     -------
#     numpy.ndarray
#         A 1D NumPy array representing g_ij, the scattering factor product
#         for each atomic pair.

#     Notes
#     -----
#     The function performs the following steps:
#     1.  Retrieves the atomic numbers zi for each unique atom.
#     2.  Assigns the corresponding zi to each atom in the pairs defined by aij.
#     3.  Calculates the product of the atomic numbers for each pair.
#     4.  Normalizes the product by the square of the average atomic number,
#         weighted by the atomic occupancies.
#     """
#     symbols_all = self.phase.model["symbol"]
#     occ = self.phase.model["occ"]

#     sfs = np.ones_like(symbols_all).astype(float)
#     fij = np.ones_like(aij).astype(float)

#     symbols = np.unique(symbols_all)
#     for atom in symbols:
#         zi = self.zi[atom]
#         sfs[np.where(symbols_all == atom)] = zi
#         fij[np.where(aij == atom)] = zi

#     gij = np.abs(fij[:, 0] * fij[:, 1])
#     gij /= np.square(np.average(sfs * occ))  # fij/<cf>2
#     return gij



# def calc_scattering_factors(self) -> NDArray[np.float64]:
#     """
#     Calculates the atomic scattering factors (Q-independent approximation).

#     This function calculates the atomic scattering factors f(Q) using the
#     approximation that they are equal to the atomic numbers (zi).  This
#     is a simplification that neglects the Q-dependence of the scattering factors.

#     Parameters
#     ----------
#     self : Simulator
#         The Simulator object, which must contain the atomic symbols and
#         atomic numbers (zi).

#     Returns
#     -------
#     numpy.ndarray
#         A 1D NumPy array of the atomic scattering factors f(Q) for each atom
#         in the structure.
#     """
#     symbols_all = self.phase.model["symbol"]
#     sfs = np.ones_like(symbols_all).astype(float)
#     symbols = np.unique(symbols_all)
#     for atom in symbols:
#         zi = self.zi[atom]
#         sfs[np.where(symbols_all == atom)] = zi
#     return sfs
