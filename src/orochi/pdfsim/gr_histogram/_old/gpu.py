import numpy as np


def calc_gr_histogram(self):
    """
    Calculates the pair distribution function g(r) histogram using GPU.

    This function calculates the pair distribution function g(r) histogram
    using GPU acceleration. It performs a direct calculation using CUDA without
    the grouping strategy employed in the CPU version.

    Parameters
    ----------
    self : Calculator
        An instance of the Calculator class for PDF calculation.

    Returns
    -------
    r : numpy.ndarray
        Array of radial distances r.
    gr : numpy.ndarray
        The calculated histogram of the pair distribution function g(r).

    Notes
    -----
    This function performs the following steps:

    1.  Calculates the radial distances `r` and `r_rij`.
    2.  Calculates the scattering factors using `calc_scattering_factors`.
    3.  Obtains atomic coordinates, translational vectors, occupancies, and
        atomic displacement parameters.
    4.  Calls the CUDA-accelerated `calc_pdf` function to compute the histogram.
    5.  Normalizes the resulting histogram.
    """
    rstep = self.r[1] - self.r[0]
    r_extension = self.num_p_peak / 2.0 / self["qmax"]
    r = np.arange(self.r[0], self.r[-1] + rstep + r_extension, rstep)

    r_rij = np.arange(0.0, r_extension, rstep)
    r_rij = np.append(-r_rij[:0:-1], r_rij)

    from ..structure_factor.no_q_depend import calc_scattering_factors
    from .histogram.pdf_no_q_depend_gpu import calc_pdf

    sf = calc_scattering_factors(self)

    xyz = self.phase.model["xyz"]
    occ = self.phase.model["occ"]
    adp = self.phase.model["adp"]
    g = self.phase.model["lattice_vector"].T
    xyz = np.dot(g, xyz.T).T
    tvs = self.translational_vectors

    gr = calc_pdf(xyz, tvs, occ, sf, adp, r, r_rij)
    gr /= r
    gr *= (self["qmax"] * self["pad"]) / np.pi
    gr /= occ.size
    return r, gr
