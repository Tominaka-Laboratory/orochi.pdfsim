import numpy as np
import scipy.interpolate as intp
from scipy.spatial import distance
from scipy.fft import dct
from orochi.base.core import OrochiBase
from .params import default
import copy


class Simulator(OrochiBase):
    r"""
    Calculates the Pair Distribution Function (PDF), G(r).

    This class simulates the PDF based on the structural information provided
    by a `Phase` object. It manages simulation parameters, including r-range,
    Q-range, instrumental effects, and radiation-specific settings. The class
    supports different particle shapes ("analytical" for bulk periodic systems
    or explicit atomic coordinates for nanostructures) and can use either CPU
    or GPU for computation.

    For "analytical" shapes (typically bulk crystals), pair information within
    a unit cell is calculated and extended using translational vectors to
    simulate the PDF over the desired r-range. For explicitly defined
    structures (e.g., nanoparticles), pairwise distances are calculated
    directly from the provided atomic coordinates.

    The PDF calculation involves several key steps:
    1.  **Pair Distance ($r_{ij}$)**: The distance between atom i and atom j.
        For periodic systems, this includes lattice translations.
    2.  **Scattering Amplitude Weighting ($g_{ij}$)**: Each pair's
        contribution is weighted by its scattering power, typically involving
        atomic form factors $f_i(Q)$ and concentrations $c_i$.
        $$
        g_{ij}(Q) = \frac{c_i c_j f_i(Q) f_j(Q)}{\langle f(Q) \rangle^2}
        $$
    3.  **Atomic Displacement Broadening**: PDF peaks are broadened due to
        atomic displacements (thermal and static disorder), modeled as a
        Gaussian convolution. The peak width $\sigma_{ij}$ is related to the
        mean square relative displacement (MSRD), often approximated from
        isotropic Atomic Displacement Parameters ($U_{iso}$).
        $$
        \sigma_{ij} \approx \sqrt{U_{iso,i} + U_{iso,j}}
        $$
    4.  **PDF Simulation ($G(r)$)**: The final reduced PDF, $G(r)$, is
        calculated as:
        $$
        G(r) = \frac{1}{r} \left( \sum_{i \neq j} \frac{c_i c_j f_i f_j}{\langle f \rangle^2} \delta(r - r_{ij}) * A(r, \sigma_{ij}) \right) - 4\pi r \rho_0
        $$
        where $A(r, \sigma_{ij})$ is the broadening function and $\rho_0$ is
        the average number density.

    Attributes
    ----------
    phase : orochi.phase.Phase
        The `Phase` object containing the structural model.
    r : numpy.ndarray
        Array of radial distances ($Å$) for the final PDF output.
    Gr : numpy.ndarray
        The final calculated reduced Pair Distribution Function, $G(r)$.
    density : numpy.ndarray
        The average atomic number density term, $4\pi r \rho_0(r)$.
    shape_function : numpy.ndarray
        A function applied to model nanoparticle shape effects.
    pairs : dict
        A dictionary storing pairwise information such as 'r_ij', 'c_ij',
        'sigma_ij', and 'symbol_ij'.
    partial_gr : dict[str, numpy.ndarray]
        A dictionary storing the partial pair distribution functions $g(r)$ for
        each unique atom pair type (e.g., ``{"Ce-Ce": ..., "Ce-O": ..., "O-O": ...}``).
    sfs : numpy.ndarray
        An array of atomic scattering factors, which can be Q-dependent
        (complex) or Q-independent (atomic numbers).
    gij : numpy.ndarray
        The pair-specific scattering weight used in the g(r) calculation.
    sigma_r_dependence : list
        A list of r-dependent factors applied to peak widths to model
        correlated motion and instrumental broadening.
    translational_vectors : numpy.ndarray
        Lattice translation vectors for simulating periodic systems.
    obs_x : numpy.ndarray or None
        Observed experimental r grid (if registered via `set_obs_data`).
    obs_y : numpy.ndarray or None
        Observed experimental G(r) (if registered via `set_obs_data`).
    """

    def __init__(self):
        """
        Initializes the Simulator class.

        Sets up default parameters, numerical precision types, and other
        instance attributes.
        """
        self._par = default.par
        self._par_key_info = default.par_key_info

        super().__init__()

        self._name = "orochi.pdfsim"
        self._id = "0"
        self._phase = None
        self._r = None
        self._Gr = None

        self._obs_x = None
        self._obs_y = None
        self._score = {}

        self.pairs: dict = {}
        """dict: A dictionary storing pairwise information such as 'r_ij', 'c_ij', 'sigma_ij', and 'symbol_ij'."""

        self.partial_gr: dict = {}
        """dict: Dictionary storing the partial pair distribution functions g(r) for each unique atomic pair."""

        self._method_dependencies = {
            "set_r": {"self": ["rmin", "rmax", "rstep", "qmax", "pad"]},
            "calc_shape_function": {
                "self": ["rmin", "rmax", "rstep", "qmax", "pad"],
                "phase": ["shape_type", "sphere", "dimension"]
            },
            "set_scattering_factors": {
                "self": ["qdepend", "qmax", "rmax", "rmax_extension", "energy", "rambda"],
                "phase": ["symbol"]
            },
            "calc_translational_vector": {
                "self": ["rmax"],
                "phase": ["pbc", "lattice", "shape_type"],
            },
            "calc_rij": {"phase": ["xyz", "lattice"]},
            "calc_sij": {"phase": ["adp", "lattice"]},
            "calc_cij": {"phase": ["occ", "lattice"]},
            "calc_aij": {"phase": ["label", "lattice"]},
            "get_unique_pair_indices": {
                "self": ["rmin", "rmax", "rstep", "qmax", "pad"],
                "phase": ["xyz", "lattice", "symbol"]},
            "calc_gij": {
                "self": ["qdepend", "qmax", "rmax", "rmax_extension", "energy", "rambda", "pad"],
                "phase": ["occ", "symbol"]},

            "calc_sigma_r_dependence": {
                "self": ["delta1","delta2", "qbroad", "sratio"],
                "phase": ["symbol", "label"],
            },
            "calc_gr": {
                "self": ["qdepend", "qmax", "rmin", "rmax", "rstep", "rmax_extension", "energy", "rambda", "pad",
                         "delta1","delta2", "qbroad", "sratio"],
                "phase": ["xyz", "lattice", "occ", "adp", "label", "symbol"],
            },
            "calc_density": {
                "self": ["rmin", "rmax", "rstep", "qmax", "pad", "qmin"],
                "phase": ["xyz", "lattice", "occ", "shape_type"],
            }
        }

    @property
    def obs_x(self):
        """numpy.ndarray or None: Observed experimental radial grid (r in Å)."""
        if getattr(self, "_obs_module", None) is not None:
            r = getattr(self._obs_module, "r", None)
            if r is not None:
                try:
                    arr = np.asarray(r, dtype=float)
                    if arr.ndim > 0 and len(arr) > 0:
                        return arr
                except Exception:
                    pass
        return self._obs_x

    @obs_x.setter
    def obs_x(self, value):
        if value is None:
            self._obs_x = None
        else:
            self._obs_x = np.asarray(value, dtype=float)

    @property
    def obs_y(self):
        """numpy.ndarray or None: Observed experimental reduced Pair Distribution Function (G(r))."""
        if getattr(self, "_obs_module", None) is not None:
            gr = getattr(self._obs_module, "Gr", None)
            if gr is None and not hasattr(self._obs_module, "Gr") and hasattr(self._obs_module, "gr"):
                gr = getattr(self._obs_module, "gr", None)
            if gr is not None:
                try:
                    arr = np.asarray(gr, dtype=float)
                    if arr.ndim > 0 and len(arr) > 0:
                        return arr
                except Exception:
                    pass
        return self._obs_y

    @obs_y.setter
    def obs_y(self, value):
        if value is None:
            self._obs_y = None
        else:
            self._obs_y = np.asarray(value, dtype=float)

    def sync_from_pdfget(self, pdf_mod, auto_range: bool = True):
        """Synchronizes Fourier truncation (qmax, qmin), radiation (rambda, energy),
        and radial grid parameters (rmin, rmax, rstep) from a PDFget instance.

        Parameters
        ----------
        pdf_mod : OrochiBase or PDFget
            Source module producing reduced PDF data.
        auto_range : bool, default True
            Whether to update radial grid range (rmin, rmax, rstep) to match observed data.

        Returns
        -------
        self : Simulator
        """
        if pdf_mod is None:
            return self

        # 1. Fourier cutoff parameters (qmax, qmin)
        for qk in ["qmax", "qmin"]:
            if hasattr(pdf_mod, "par") and qk in pdf_mod.par:
                val = pdf_mod[qk]
                if val is not None:
                    try:
                        self[qk] = float(np.ravel(val)[0])
                    except Exception:
                        pass

        # 2. Radiation parameters (rambda, energy)
        if hasattr(pdf_mod, "par"):
            if "rambda" in pdf_mod.par and pdf_mod["rambda"] is not None:
                try:
                    self["rambda"] = float(np.ravel(pdf_mod["rambda"])[0])
                except Exception:
                    pass
            elif "energy" in pdf_mod.par and pdf_mod["energy"] is not None:
                try:
                    self["energy"] = float(np.ravel(pdf_mod["energy"])[0])
                except Exception:
                    pass

        # 3. Radial grid parameters
        if auto_range:
            r_arr = getattr(pdf_mod, "r", None)
            if r_arr is not None and len(r_arr) > 1:
                try:
                    r_clean = np.asarray(r_arr, dtype=float).flatten()
                    self["rmin"] = max(0.0, float(np.min(r_clean)))
                    self["rmax"] = float(np.max(r_clean))
                    diffs = np.diff(r_clean)
                    if len(diffs) > 0:
                        dr = float(np.round(np.mean(diffs), 4))
                        if dr > 0:
                            self["rstep"] = dr
                except Exception:
                    pass
            elif hasattr(pdf_mod, "par"):
                for rk in ["rmin", "rmax"]:
                    if rk in pdf_mod.par and pdf_mod[rk] is not None:
                        try:
                            self[rk] = float(np.ravel(pdf_mod[rk])[0])
                        except Exception:
                            pass

        return self

    def set_obs_data(self, obs_x, obs_y=None, auto_range: bool = True):
        """
        Set experimental PDF observation data for refinement and model evaluation.

        Parameters
        ----------
        obs_x : array_like, str, or OrochiBase
            - 1D array of experimental radial grid points (r in Å), OR
            - File path to a two-column text / .gr file (column 0: r, column 1: G(r)), OR
            - An Orochi module producing r and Gr (e.g. `PDFget` instance).
        obs_y : array_like, optional
            1D array of experimental G(r) values. Required when `obs_x` is an array.
        auto_range : bool, default True
            If True, automatically adjusts the simulator's `rmin`, `rmax`, and
            `rstep` to align with the observed data grid.

        Returns
        -------
        self : Simulator
            Returns self to enable method chaining.

        Examples
        --------
        >>> # Pass an Orochi module (e.g. PDFget) directly — links by reference!
        >>> simulator.set_obs_data(pdfget)
        >>>
        >>> # Pass arrays directly
        >>> simulator.set_obs_data(r, Gr)
        >>>
        >>> # Load from a .gr or text file
        >>> simulator.set_obs_data("sample_reduced.gr")
        """
        import os

        # 1. Handle OrochiBase / PDFget module instance (or any object with .r and .Gr/.gr)
        if hasattr(obs_x, "r") and (hasattr(obs_x, "Gr") or hasattr(obs_x, "gr")):
            self._obs_module = obs_x
            self.sync_from_pdfget(obs_x, auto_range=auto_range)
            r = getattr(obs_x, "r", None)
            gr_val = getattr(obs_x, "Gr", None)
            if gr_val is None and not hasattr(obs_x, "Gr") and hasattr(obs_x, "gr"):
                gr_val = getattr(obs_x, "gr", None)
            if r is not None and gr_val is not None:
                try:
                    x = np.asarray(r, dtype=float)
                    y = np.asarray(gr_val, dtype=float)
                    if x.ndim > 0 and y.ndim > 0 and len(x) > 0 and len(y) > 0 and len(x) == len(y):
                        self._obs_x = x.flatten()
                        self._obs_y = y.flatten()
                except Exception:
                    pass
            return self

        # 2. Handle file path input (e.g., .gr or .txt file)
        self._obs_module = None
        if isinstance(obs_x, str) and (obs_y is None or isinstance(obs_y, (bool, dict))):
            if not os.path.exists(obs_x):
                raise FileNotFoundError(f"Observed data file not found: {obs_x}")
            # Parse header metadata if available (qmax, qmin, wavelength)
            try:
                with open(obs_x, "r", encoding="utf-8", errors="ignore") as f:
                    for _ in range(50):
                        line = f.readline()
                        if not line:
                            break
                        line_s = line.strip().lstrip("#").strip()
                        if "=" in line_s:
                            parts = line_s.split("=")
                            k_hdr = parts[0].strip().lower()
                            v_hdr_str = parts[1].strip().split()[0]
                            try:
                                v_hdr = float(v_hdr_str)
                                if k_hdr == "qmax" and "qmax" in self.par:
                                    self["qmax"] = v_hdr
                                elif k_hdr == "qmin" and "qmin" in self.par:
                                    self["qmin"] = v_hdr
                                elif (k_hdr in ("wavelength", "rambda")) and "rambda" in self.par:
                                    self["rambda"] = v_hdr
                            except Exception:
                                pass
                        elif not line.startswith("#"):
                            break
            except Exception:
                pass
            raw = np.genfromtxt(obs_x, comments="#")
            if raw.ndim == 1 or raw.shape[1] < 2:
                raise ValueError(f"Expected at least 2 columns (r, G(r)) in file {obs_x}")
            x = raw[:, 0]
            y = raw[:, 1]
        # 3. Handle array inputs
        else:
            if obs_y is None:
                raise ValueError("obs_y must be provided when obs_x is an array.")
            x = np.asarray(obs_x, dtype=float)
            y = np.asarray(obs_y, dtype=float)

        # Ensure 1D arrays
        if x.ndim != 1 or y.ndim != 1:
            x = x.flatten()
            y = y.flatten()

        if len(x) != len(y):
            raise ValueError(
                f"Length mismatch: obs_x has length {len(x)}, obs_y has length {len(y)}"
            )
        if len(x) == 0:
            raise ValueError("Observed data arrays are empty.")

        # 4. Optional auto-range and grid step alignment
        if auto_range and len(x) > 1:
            x_min = float(np.min(x))
            x_max = float(np.max(x))
            self["rmin"] = max(0.0, x_min)
            self["rmax"] = x_max
            diffs = np.diff(x)
            if len(diffs) > 0:
                dr = float(np.round(np.mean(diffs), 4))
                if dr > 0:
                    self["rstep"] = dr

        self._obs_x = x
        self._obs_y = y
        return self

    def __setitem__(self, key, value):
        r"""
        Sets a parameter value, with special handling for energy/wavelength.

        If 'energy' or 'rambda' (wavelength) is set, its coupled counterpart
        is automatically updated using the relation $E = hc/\lambda$.

        Parameters
        ----------
        key : str or tuple
            The key of the parameter to set.
        value : any
            The value to assign to the parameter.

        Returns
        -------
        tuple
            A tuple representing the processed key, attribute, and indices.
        """
        key, attribute, index1, index2 = super().__setitem__(key, value)

        if key == "energy":
            self.set_rambda_from_energy()

        elif key == "rambda":
            self.set_energy_from_rambda()
        return key, attribute, index1, index2

    def __call__(self, force: bool = False):
        """
        Executes the full PDF simulation pipeline.

        This method orchestrates the sequence of calculations:
        1.  Processes the input `Phase` object.
        2.  Sets up the radial distance grid (`self._r`).
        3.  Calculates the nanostructure shape function.
        4.  Initializes atomic scattering factors.
        5.  Calculates translational vectors if the shape is "analytical".
        6.  Computes all pairwise information (`rij`, `cij`, `sij`, `aij`).
        7.  Groups pairs by type and computes scattering weights (`gij`).
        8.  Calculates r-dependent peak width corrections.
        9.  Computes the pair correlation function `g(r)`.
        10. Computes the number density term.
        11. Assembles the final `G(r)`.

        Parameters
        ----------
        force : bool, optional
            If True, resets internal state to 'INIT' to force full re-evaluation
            of all geometric pairs and shape envelopes without skipping cached steps,
            by default False.
        """
        if force:
            self._state = "INIT"
            if hasattr(self, "phase") and self.phase is not None and hasattr(self.phase, "_state"):
                self.phase._state = "INIT"
        try:
            super().__call__(force=force)
            if hasattr(self, "phase") and self.phase is not None:
                if hasattr(self.phase, "run"):
                    self.phase.run(force=force)
                else:
                    self.phase(force=force) if hasattr(self.phase, "__call__") else self.phase()
            self.set_r()
            self.calc_shape_function()

            self.set_scattering_factors()
            self.calc_translational_vector()

            self.calc_rij()
            self.calc_sij()
            self.calc_cij()
            self.calc_aij()

            self.get_unique_pair_indices()
            self.calc_gij()
            self.calc_sigma_r_dependence()
            self.calc_gr()
            self.calc_density()

            self.calc_Gr()
            self._state = "DONE"
        except Exception as e:
            if self._state == "DONE":
                self._state = "INIT"
                self.__call__(force=True)
            else:
                self._state = "ERROR"
                if hasattr(self, "_logger") and self._logger:
                    self._logger.error(f"PDF simulation failed: {e}", exc_info=True)
                raise

    def run(self, force: bool = False):
        """Explicit execution entry point for user scripts and interactive environments.

        Parameters
        ----------
        force : bool, optional
            If True, forces full recalculation by bypassing cached state, by default False.
        """
        return self.__call__(force=force)
                

    def __add__(self, other):
        """
        Adds the G(r) of this simulation to another Simulator's G(r).

        This operation requires that both simulators have been executed and
        that their `r` grids are identical.

        Parameters
        ----------
        other : Simulator
            Another `Simulator` instance whose `Gr` attribute will be added.

        Returns
        -------
        numpy.ndarray
            The element-wise sum of `self.Gr` and `other.Gr`.

        Raises
        ------
        NotImplementedError
            If `other` is not a `Simulator` instance or if the `r` grids
            do not match.
        ValueError
            If `Gr` has not been calculated for one or both instances.
        """
        if isinstance(other, Simulator) and hasattr(other, 'Gr') and hasattr(self, 'Gr'):
            if self.Gr is not None and other.Gr is not None:
                if self.r.shape == other.r.shape and np.allclose(self.r, other.r):
                    return self.Gr + other.Gr
                else:
                    self._logger.error("Cannot add G(r) from Simulators with different r-grids.")
                    raise NotImplementedError("Addition of G(r) with non-matching r-grids is not implemented.")
            else:
                self._logger.error("One or both G(r) attributes are None. Cannot add.")
                raise ValueError("Cannot add G(r) if one or both are None.")
        return NotImplemented

    def set_r(self):
        r"""
        Defines the internal radial distance grid `self._r` for calculation.

        The step size is determined by the Nyquist criterion based on
        `self["qmax"]` and a padding factor:
        $r_{step} = \pi / (Q_{max} \cdot \text{pad})$.
        The grid extends from this step size up to `self["rmax"]` plus an
        extension factor `self["rmax_extension"]`. This internal grid, `self._r`,
        is used for high-resolution calculations before final interpolation
        onto the output `self.r` grid.
        """
        rstep = np.pi / (self["qmax"] * self["pad"])
        extension = self["rmax_extension"]
        self._r = np.arange(rstep, self["rmax"] + extension, rstep)

        self.calc_min = self._r[0] - extension
        self.calc_max = self._r[-1] + extension

        if self.calc_min <= 0:
            self.calc_min = 0

    @property
    def r(self):
        """
        Gets the radial distance grid for output.
        """
        r = np.arange(self["rmin"], self["rmax"] + self["rstep"], self["rstep"])
        return r

    def calc_shape_function(self):
        r"""
        Calculates the particle shape function $S(r)$.

        This function modifies the PDF to account for finite particle size.
        - If `phase["shape_type"]` is "analytical" and a valid spherical
          diameter is provided in `phase["sphere"]`, an analytical shape
          function for a sphere is calculated:
          $S(r) = 1 - 1.5(r/D) + 0.5(r/D)^3$ for $r \le D$.
        - Otherwise, $S(r) = 1.0$ for all r, indicating no shape effect.

        The result is stored in `self.shape_function`.
        """
        if self.phase["shape_type"] == "analytical":
            try:
                diameters_param = self.phase["sphere", "value"]
                dimension = self.phase["sphere", "dimension"]
                if diameters_param is not None:
                    diameters = np.array(diameters_param, dtype=float)
                    if diameters.size > 0 and np.all(diameters > 0):
                        avg_diameter = np.average(diameters)
                        r_eff = np.maximum(self.r, 1e-9)
                        ratio = r_eff / avg_diameter

                        value = 1.0 - 1.5 * ratio + 0.5 * (ratio ** 3.0)

                        if dimension != 3:
                            r_for_dim_power = np.maximum(r_eff, 1e-9) if (3.0 - dimension) > 0 else r_eff
                            value /= (r_for_dim_power ** (3.0 - dimension))

                        current_shape_function = np.where(r_eff <= avg_diameter, value, 0.0)
                        self.shape_function = np.clip(current_shape_function, 0.0, 1.0)
                        return
            except KeyError:
                pass

        self.shape_function = np.ones_like(self.r)

    def set_scattering_factors(self):
        """
        Initializes atomic scattering factors for all atoms in the model.

        Populates dictionaries for atomic numbers (`self.zi`), Q-dependent
        form factors (`self.f0s`), and anomalous scattering factors
        (`self.f1f2s`).

        It then computes the scattering factor array `self.sfs`. If
        `self["qdepend"]` is True, this array contains Q-dependent complex
        scattering factors. Otherwise, it contains Q-independent atomic
        numbers (Z).
        """
        symbols = self.phase.model["symbol"]
        self.zi = self.phase.get_atom_info_dicionary(
            symbols, key="electron"
        )
        self.f0s = self.phase.get_atom_info_dicionary(
            symbols, key="f0"
        )
        self.f1f2s = self.phase.get_atom_info_dicionary(
            symbols, key="f1f2"
        )

        symbols_all = self.phase.model["symbol"]
        symbols = np.unique(symbols_all)
        if self["qdepend"]:
            symbols = self.phase.model["symbol"]
            qstep = np.pi / (self["rmax"] + self["rmax_extension"])
            q = np.arange(0, self["qmax"] + qstep, qstep)
            sfs = np.zeros((symbols_all.size, q.size)) + 0.0j
            for atom in symbols:
                f0 = self.f0s[atom](q)
                f1, f2 = self.f1f2s[atom](self["energy"])
                sf = f0 + f1 + 1.0j * f2
                sfs[np.where(symbols_all == atom)] = sf
            self.sfs = sfs
        else:
            sfs = np.ones_like(symbols_all).astype(float)
            for atom in symbols:
                zi = self.zi[atom]
                sfs[np.where(symbols_all == atom)] = zi
            self.sfs = sfs

    def calc_translational_vector(self):
        """
        Calculates lattice translational vectors for periodic systems.

        If `phase["shape_type"]` is "analytical", this method generates a set
        of translational vectors to construct a supercell large enough to
        include all pairs within `self["rmax"]`. The vectors are stored in
        `self.translational_vectors`.
        """
        if self.phase["shape_type"] != "analytical":
            return

        radius = float(self["rmax"])
        g = self.phase.model["lattice_vector"].T

        x, y, z = np.meshgrid(
            np.arange(-1, 2), np.arange(-1, 2), np.arange(-1, 2)
        )
        xyz = np.concatenate(
            [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], axis=-1
        )
        xyz_base = np.dot(g, xyz.T).T
        xyz = xyz[(xyz.shape[0] - 1) // 2 + 1:]
        xyz_car = np.dot(g, xyz.T).T
        r = np.linalg.norm(xyz_car, axis=-1)

        mag = np.abs(np.floor(radius / r).astype(int)) + 1
        xyz = xyz * mag[:, np.newaxis] * self.phase.model["pbc"]
        x, y, z = np.amax(np.abs(xyz), axis=0)

        x, y, z = np.meshgrid(
            np.arange(-x, x + 1),
            np.arange(-y, y + 1),
            np.arange(-z, z + 1),
        )
        xyz = np.concatenate(
            [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], axis=-1
        ).astype(float)

        xyz = np.dot(g, xyz.T).T
        r = distance.cdist(xyz, xyz_base)
        use = np.nonzero(np.sum(np.where(r <= radius, 1.0, 0.0), axis=-1))

        self.translational_vectors = np.round(xyz[use], 7)

    def calc_rij(self):
        r"""
        Calculates all unique pairwise distances ($r_{ij}$).

        - If `phase["shape_type"]` is "analytical", it computes distances
          between atoms in the unit cell and all translated lattice points.
        - Otherwise (for explicit, non-periodic structures), it directly
          calculates all pairwise distances from the given atomic coordinates.

        The resulting flat array of distances is stored as `self.pairs["r_ij"]`.
        """

        def get_rij_from_xyz(xyz):
            p = xyz.shape[0]
            y, x = np.meshgrid(np.arange(p), np.arange(p))
            rij = xyz[np.newaxis, :] - xyz[:, np.newaxis]
            rij = np.array(rij[y >= x], dtype=np.float64)
            return rij

        xyz = self.phase.model["xyz"]
        g = self.phase.model["lattice_vector"].T

        if self.phase["shape_type"] == "analytical":
            xyz_car = np.dot(g, xyz.T).T
            rij = get_rij_from_xyz(xyz_car)
            rij = distance.cdist(
                rij, self.translational_vectors
            ).flatten()
        else:
            xyz_car = np.dot(g, xyz.T).T
            pair_vectors = get_rij_from_xyz(xyz_car)
            rij = np.linalg.norm(pair_vectors, axis=-1).flatten()

        rij = np.round(rij, 7)
        self.sim_range = np.where((rij > self.calc_min) & (rij < self.calc_max))
        self.pairs["r_ij"] = rij

    def calc_cij(self):
        r"""
        Calculates concentration products ($c_i c_j$) for each pair.

        This method computes the product of site occupancies for each atom
        pair. A multiplicity factor of 2.0 is applied to non-identical pairs
        (i != j). For analytical shapes, the unit cell products are tiled to
        match the supercell.

        The flat array of products is stored as `self.pairs["c_ij"]`.
        """

        def get_cij_from_occ(occ):
            p = occ.shape[0]
            y, x = np.meshgrid(np.arange(p), np.arange(p))
            cij = occ[np.newaxis, :] * occ[:, np.newaxis]
            cij[y > x] *= 2.0
            cij = np.array(cij[y >= x], dtype=np.float64)
            return cij

        occ = self.phase.model["occ"]
        if self.phase["shape_type"] == "analytical":
            cij = get_cij_from_occ(occ)
            p = self.translational_vectors.shape[0]
            cij = np.tile(cij, (p, 1)).reshape(p, -1).T.flatten()
        else:
            cij = get_cij_from_occ(occ)

        cij = np.round(cij, 7)
        self.pairs["c_ij"] = cij

    def calc_sij(self):
        r"""
        Calculates mean-square relative displacements ($\sigma_{ij}$) for pairs.

        It computes $\sigma_{ij} = \sqrt{U_{iso,i} + U_{iso,j}}$ for each pair,
        representing the initial broadening due to atomic displacements. For
        analytical shapes, these values are tiled for the supercell.

        The flat array is stored as `self.pairs["sigma_ij"]`.
        """

        def get_uij_from_adp(adp):
            p = adp.shape[0]
            y, x = np.meshgrid(np.arange(p), np.arange(p))
            sij = adp[np.newaxis, :] + adp[:, np.newaxis]
            sij = np.sqrt(sij)
            sij = np.array(sij[y >= x], dtype=np.float64)
            return sij

        adp = self.phase.model["adp"]
        if self.phase["shape_type"] == "analytical":
            sij = get_uij_from_adp(adp)
            p = self.translational_vectors.shape[0]
            sij = np.tile(sij, (p, 1)).reshape(p, -1).T.flatten()
        else:
            sij = get_uij_from_adp(adp)

        sij = np.round(sij, 7)
        self.pairs["sigma_ij"] = sij

    def calc_aij(self):
        r"""
        Determines atomic symbols for each pair (e.g., ('Fe', 'O')).

        Identifies the symbols of the two atoms for each pairwise distance. For
        analytical shapes, the unit cell pairs are tiled to match the supercell.

        A sorted array of symbol pairs is stored as `self.pairs["symbol_ij"]`.
        """

        def get_at_from_symbols(symbols):
            p = symbols.shape[0]
            y, x = np.meshgrid(np.arange(p), np.arange(p))
            m = np.tile(symbols, (p, 1))
            ai = m[y >= x]
            aj = m.T[y >= x]
            return ai, aj

        symbols = self.phase.model["symbol"]
        if self.phase["shape_type"] == "analytical":
            ai, aj = get_at_from_symbols(symbols)
            p = self.translational_vectors.shape[0]
            ai = np.tile(ai, (p, 1)).reshape(p, -1).T.flatten()
            aj = np.tile(aj, (p, 1)).reshape(p, -1).T.flatten()
        else:
            ai, aj = get_at_from_symbols(symbols)

        aij = np.stack((ai, aj), axis=1)
        self.pairs["symbol_ij"] = np.sort(aij, axis=-1)

    def get_unique_pair_indices(self):
        """
        Identifies and groups unique pair types (e.g., all 'Fe-Fe' pairs).

        This method processes the list of all pairs to find unique pair types
        (like 'Fe-O', 'Si-Si', etc.). It stores the indices and boundaries for
        these groups, which allows for efficient, vectorized calculations on a
        per-pair-type basis later in the simulation.
        """

        aij = self.pairs["symbol_ij"][self.sim_range]
        self.pair_order = np.lexsort((aij[:, 1], aij[:, 0]))
        aij = aij[self.pair_order]

        unique_aij, indices, counts_aij = np.unique(
            aij, return_index=True, return_counts=True, axis=0
        )

        index_sort = np.argsort(indices)
        index_aij = indices[index_sort]
        counts_aij = counts_aij[index_sort]
        self.pair_group_bounds = index_aij + counts_aij

        self.unique_pair_indices = indices
        aij = aij[self.unique_pair_indices]
        self.pair_name = unique_aij[:, 0].astype(object) + "-"+ unique_aij[:, 1].astype(object)
        self.unique_aij = unique_aij

    def calc_gij(self):
        """
        Calculates the pair-specific scattering weight term, g_ij.

        This term weights the contribution of each pair type to the total PDF.
        - If `qdepend` is True, it calculates the Q-dependent weights, normalizes
          by the average scattering factor, and performs a Discrete Cosine
          Transform (DCT) to move from Q-space to r-space.
        - If `qdepend` is False, it calculates a simple weight based on the
          product of atomic numbers normalized by the average.
        The result is stored in `self.gij`.
        """
        if self["qdepend"]:
            symbols = np.unique(self.unique_aij)
            symbols_all = self.phase.model["symbol"]
            num_unique_atoms1 = self.unique_aij.shape[0]
            num_unique_atoms2 = self.unique_aij.shape[1]
            fij = np.zeros((num_unique_atoms1, num_unique_atoms2, self.sfs.shape[1]), dtype=complex)

            for atom in symbols:
                id = np.where(symbols_all == atom)
                id2 = np.where(self.unique_aij == atom)
                fij[id2] = self.sfs[id][0]

            gij = fij[:, 0] * fij[:, 1]
            occ = self.phase.model["occ"]
            sq_ave_cf = np.square(np.average(self.sfs * np.reshape(occ, (-1, 1)), axis=0))
            gij = np.abs(gij / sq_ave_cf)

            qn = gij.shape[1] * self["pad"]
            gij = dct(gij, n=qn, type=2, norm=None, axis=1) / (2.0 * qn)
            self.gij = np.append(gij[:, :0:-1], gij, axis=-1)
        else:
            fij = np.ones_like(self.unique_aij).astype(float)
            symbols = np.unique(self.unique_aij)
            for atom in symbols:
                zi = self.zi[atom]
                fij[np.where(self.unique_aij == atom)] = zi

            gij = np.abs(fij[:, 0] * fij[:, 1])
            sq_ave_cf = np.square(np.average(self.sfs * self.phase.model["occ"]))
            self.gij = gij / sq_ave_cf

    def calc_sigma_r_dependence(self):
        r"""
        Calculates r-dependent modifications to PDF peak widths.

        This method applies corrections to the base MSRD ($\sigma_{ij}$) to
        model effects of correlated atomic motion and instrumental resolution.
        It uses the PDFfit2 model:
        $$
        \text{factor} = \sqrt{1 - \frac{\delta_1}{r} - \frac{\delta_2}{r^2}
        + (Q_{broad} \cdot r)^2}
        $$
        Additionally, it can apply specific scaling factors for designated
        element pairs within certain distance ranges, as defined in the
        `sratio` parameter. The final multiplicative factors are stored
        in `self.sigma_r_dependence`.
        """
        term_inside_sqrt = (1.0
                            - self["delta1"] / self._r
                            - self["delta2"] / np.square(self._r)
                            + np.square(self["qbroad"] * self._r))
        general_width_factor = np.sqrt(np.maximum(term_inside_sqrt, 0.0))

        peak_width = general_width_factor
        self.sigma_r_dependence = []

        rcut = np.array(self["sratio", "value"])
        if "pair" in self["sratio"] and "range" in self["sratio"] and \
                (not np.all(rcut == 1.0)):
            applied_range = np.array(self["sratio", "range"])
            pairs = np.array(self["sratio", "pair"])

            for n, name in enumerate(self.pair_name):
                aij = self.unique_aij[n]
                ai = aij[0]
                aj = aij[1]
                specific_pair_factors = np.ones_like(self._r)

                for m, (pair1_target, pair2_target) in enumerate(pairs):
                    if ((ai == pair1_target) & (aj == pair2_target)) or ((ai == pair2_target) & (aj == pair1_target)):
                        mask = np.where((self._r > applied_range[m][0]) & (self._r < applied_range[m][1]))
                        specific_pair_factors[mask] *= rcut[m]

                self.sigma_r_dependence.append(peak_width * specific_pair_factors)
        else:
            self.sigma_r_dependence = [peak_width] * len(self.pair_name)

    def calc_gr(self):
        """
        Calculates the pair correlation function g(r).

        This is the core computational step. It iterates through each unique
        pair type, calculates its contribution to the g(r) histogram, and
        applies the appropriate broadening and scattering weight factors.

        - If `qdepend` is False, a sinc convolution is applied to account for
          Qmax termination effects.
        - If `qdepend` is True, a custom convolution is performed using the
          r-space kernel derived from the DCT in `calc_gij`.

        The individual "partial" g(r) functions are summed and interpolated
        to the final output grid `self.gr`.
        """
        from .gr_histogram.histogram.cpu import gr_for_one_pair_type
        from .gr_histogram.profile.cpu import sinc_convolution, custom_kernel_convolution

        self.split_pairs = {}
        for key, value in self.pairs.items():
            ordered_value = value[self.sim_range][self.pair_order]
            split_value = np.split(ordered_value, self.pair_group_bounds[:-1])
            self.split_pairs[key] = split_value

        partial_gr = {}
        for n, rij in enumerate(self.split_pairs["r_ij"]):
            cij = self.split_pairs["c_ij"][n]
            sij = self.split_pairs["sigma_ij"][n]
            sigma_r_dependence = self.sigma_r_dependence[n]

            gr = gr_for_one_pair_type(self._r, rij, cij, sij, sigma_r_dependence)
            gr *= (self["qmax"] * self["pad"]) / np.pi
            gr /= self.phase.model["occ"].size
            gr /= self._r

            if not self["qdepend"]:
                gr *= self.gij[n]
                gr = sinc_convolution(gr, self._r, self["qmax"])
            else:
                kernel = self.gij[n]
                gr = custom_kernel_convolution(gr, self._r, kernel)

            partial_gr[self.pair_name[n]] = gr

        self.partial_gr = partial_gr
        self._gr = np.sum(list(partial_gr.values()), axis=0)
        self.gr = intp.interp1d(self._r, self._gr,
                                  fill_value="extrapolate",
                                  bounds_error=False)(self.r)

    def calc_density(self):
        r"""
        Calculates the average number density term.

        - For a 3D periodic "analytical" system, the density is calculated
          from the cell volume and number of atoms: $\rho_0 = N/V$. The term is
          then $4\pi r \rho_0$.
        - For nanostructures or non-3D-periodic systems, the density term is
          derived from the low-r behavior of g(r), effectively by applying a
          low-pass filter determined by `qmin` via a sinc convolution.

        The result is stored in `self.density`.
        """
        if (self.phase["shape_type"] == "analytical") and (np.sum(self.phase["pbc"]) == 3):
            g = self.phase.model["lattice_vector"].T
            V = np.linalg.det(g)
            Np = np.sum(self.phase.model["occ"])

            for name, subphase in self.phase.subphase:
                subphase()
                Np += np.sum(subphase.model["occ"])

            rho0 = Np / V
            if self["simulation_type"] == "pdf":
                density = 4.0 * np.pi * rho0 * self.r
            elif self["simulation_type"] == "epf":
                density = 4.0 * np.pi * rho0 * self.r ** 2.0
            else:
                density = np.zeros_like(self.r)
            self.density = density
        else:
            from .gr_histogram.profile.cpu import sinc_convolution
            density = sinc_convolution(self._gr, self._r, self["qmin"])
            self.density = intp.interp1d(self._r, density,
                                           fill_value="extrapolate",
                                           bounds_error=False)(self.r)

    def calc_Gr(self):
        r"""
        Calculates the final scaled reduced Pair Distribution Function, G(r).

        This method assembles the final G(r) by combining the calculated
        components according to the formula:
        $$
        G(r) = (g(r) - \text{density}) \cdot S(r) \cdot R(r) \cdot \text{scale}
        $$
        where:
        - $g(r)$ is `self.gr` (the pair correlation function).
        - `density` is `self.density` (the number density term).
        - $S(r)$ is `self.shape_function` (particle shape damping).
        - $R(r)$ is a resolution damping function from `self["qdamp"]`.
        - `scale` is the overall `self["scale"]` factor.

        The result is stored in `self.Gr`.
        """
        resolution = np.exp(-1.0 * np.square(self.r * self["qdamp"]) / 2.0)
        self.Gr = (self.gr - self.density) * self.shape_function * resolution * self["scale"]

    def set_N_indep(self):
        """
        Calculate and set the number of statistically independent data points (N_indep).

        This method computes N_indep based on the Nyquist-Shannon sampling theorem using 
        the current fitting range (rmin to rmax) and the maximum scattering vector (Qmax).
        The calculated value is stored in `self["N_indep"]`.

        Formula:
            N_indep = (rmax - rmin) * Qmax / pi

        Notes:
            - **Physical Meaning**: Raw experimental PDF profile points on a fine r-grid 
            are statistically correlated. N_indep represents the true number of 
            physically independent observations constrained by the experimental resolution (Qmax).
            - **Model Selection & Overfitting Prevention**: N_indep is essential for 
            degree-of-freedom corrections (e.g., Adjusted Rw, BIC). Using N_indep instead 
            of the raw number of grid points prevents the optimizer from artificially 
            favoring low-symmetry models with excessive parameters (e.g., overfitting to P1).

        Raises:
            KeyError: If 'rmax', 'rmin', or 'Qmax' are not set in the simulator settings.

        Example:
            >>> simulator["rmin"] = 1.0
            >>> simulator["rmax"] = 15.0
            >>> simulator["Qmax"] = 25.0
            >>> simulator.set_N_indep()
            >>> print(simulator["N_indep"])
            111.40846016432675
        """
        
        self["N_indep"] = (self["rmax"] - self["rmin"]) * self["qmax"] / np.pi



    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value
        self["nest_modules"] = ["phase"]

    def save_file(self, filename=None):
        """
        Save calculated PDF profiles and partial pair correlations to CSV files.

        Parameters
        ----------
        filename : str, optional
            Base path and filename without extension. If None, defaults to
            ``os.path.join(self.work_dir, self.study_name)``. Saves two files:
            ``<filename>.csv`` containing total G(r) and ``<filename>_partial.csv``
            containing partial pair contributions.
        """
        import pandas as pd
        import os

        if filename is None:
            filename = os.path.join(self.work_dir, self.study_name)
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        data = {}
        data["r"] = self._r
        data.update(self.partial_gr)
        df = pd.DataFrame(data)
        df.to_csv(filename+"_partial.csv", index=False)

        df = pd.DataFrame({"r":self.r, "Gr":self.Gr})
        df.to_csv(filename+".csv", index=False)

    def explain(self, *args, **kwargs):
        """
        Generate diagnostic summary and LLM-friendly markdown report.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to explain submodule.
        **kwargs : dict
            Keyword arguments passed to explain submodule.

        Returns
        -------
        str
            A structured markdown report of current simulation state.
        """
        self()
        from .submodules.lang_model import explain
        return explain(self, *args, **kwargs)

    def explore_subgroup_pathways(self, 
                            optimizer: dict = None,
                            nest_optimizer: dict = None,
                            top_k=5,
                            output="text",
                            **kwargs
                            ):
        """
        Explore subgroup transition pathways and screen viable candidates 
        starting from a higher-symmetry structure using two-tier nested optimization.

        This function first converts the current structure to a higher-symmetry state 
        (`make_higher_symmetry_structure`). It then performs an outer optimization to explore 
        subgroup transition paths (Hall numbers) in a dynamic parameter space, while executing 
        inner nested refinements for structural parameters (lattice, coordinates, ADPs) against 
        experimental PDF data in a fixed-dimension continuous space.

        Parameters
        ----------
        use_atoms : dict, optional
            Dictionary for used atoms for exploration.
            Keys:
                - site_indices (list, optional): List of site indices to optimize. If None, all sites are considered.
                - element_types (list, optional): List of element symbols (e.g., ['Ru']) to restrict coordinate optimization. 
                    *RECOMMENDED STRATEGY*: For initial/broad subgroup exploration under strong Z-contrast 
                    (e.g., Ru vs O), restrict this to heavy scattering elements (e.g., ["Ru"]). This stabilizes 
                    the search space by focusing on the primary scattering framework and avoiding light-atom artifacts.
            Example: {"site_indices": None, "element_types": ["Ru"]}
        lattice_parameters : dict, optional
            Dictionary for lattice constant refinement limits.
            Keys:
                - a, b, c (float): Maximum allowed change for axis lengths (in Å). Set to 0.0 to fix.
                - alpha, beta, gamma (float): Maximum allowed change for lattice angles (in degrees). Set to 0.0 to fix.
            Example: {"a": 0.3, "b": 0.3, "c": 0.0, "alpha": 0.0, "beta": 0.0, "gamma": 0.0}
        atomic_coordinates : dict, optional
            Dictionary for fractional atomic coordinate refinement.
            Keys:
                - site_indices (list, optional): List of site indices to optimize. If None, all sites are considered.
                - element_types (list, optional): List of element symbols (e.g., ['Ru']) to restrict coordinate optimization. 
                    *RECOMMENDED STRATEGY*: For initial/broad subgroup exploration under strong Z-contrast 
                    (e.g., Ru vs O), restrict this to heavy scattering elements (e.g., ["Ru"]). This stabilizes 
                    the search space by focusing on the primary scattering framework and avoiding light-atom artifacts.
                - max_change (float): Maximum allowed displacement in fractional units.
            Example: {"site_indices": None, "element_types": ["Ru"], "max_change": 0.1}
        wyckoff_positions : dict, optional
            Dictionary for exploring Wyckoff position candidate assignments.
            Keys:
                - site_indices (list, optional): List of site indices to explore. If None, all sites are considered.
                - element_types (list, optional): List of element symbols to restrict Wyckoff search.
            Example: {"site_indices": None, "element_types": None}
        particle_size : dict, optional
            Dictionary for nanoparticle morphology refinement.
            Keys:
                - min (float): Minimum nanoparticle diameter (in Å). Should be > 5.0.
                - max (float): Maximum nanoparticle diameter (in Å).
            Example: {"min": 10.0, "max": 100.0}
        element_adps : dict, optional
            Dictionary for Atomic Displacement Parameters (ADPs / B-iso).
            Keys:
                - elements (list, optional): List of element symbols (e.g., ['Ru', 'O']). If None, all elements are refined.
                - max_change (float): Maximum allowed change from current ADP values.
            Example: {"elements": ['Ru', 'O'], "max_change": 0.01}
        simulation_parameters : dict, optional
            Dictionary for instrumental and simulation parameters.
            Keys:
                - scale (float): Max change for intensity scale factor. Set to 0.0 to fix.
                - delta1, delta2 (float): Max change for peak broadening. Set to 0.0 to fix.
                - qbroad, qdamp (float): Max change for instrumental factors. Set to 0.0 to fix.
            Example: {"scale": 0.1, "qdamp": 0.0, "qbroad": 0.0}
        optimizer : dict, optional
            Configuration for the outer-loop pathway search in a dynamic parameter space.
            Keys:
                - type (str): Optimization framework ('bayes' or 'lls'). Defaults to 'bayes'.
                - fit_min (float): Minimum r-value (Å) for PDF fitting. Defaults to 0.1.
                - fit_max (float): Maximum r-value (Å) for PDF fitting. Defaults to 15.0.
                - n_trials (int): Number of outer pathway exploration trials. Defaults to 60.
                - sampler (str): Optuna sampling algorithm ('TPE' or 'CMA'). Defaults to 'TPE'. 
                                'TPE' is strongly recommended for outer dynamic tree-structured search.
                - study_name (str, optional): Optuna study name for database persistence. Set a unique name to 
                                          save or resume/continue trials in subsequent runs.
            Example: {"type": "bayes", "fit_min": 0.1, "fit_max": 15.0, "n_trials": 60, "sampler": "TPE", "study_name": "subgroup_transformation"}
        nest_optimizer : dict, optional
            Configuration for inner-loop structural parameter refinement in continuous space.
            Keys: Same structure as `optimizer`.
                - sampler (str): Defaults to 'TPE'. 'CMA' is highly recommended for inner coordinate 
                                refinement due to its superior handling of strongly correlated parameters 
                                (e.g., polyhedral coordination distortions).
                - study_name (str, optional): Optuna study name for inner-loop trials.
                - n_trials (int): Recommended to set higher (e.g., 300) when using 'CMA' to allow full convergence.
            Example: {"type": "bayes", "fit_min": 0.1, "fit_max": 15.0, "n_trials": 300, "sampler": "CMA", "study_name":None}
        top_k : int, optional
            Number of top candidate pathways to include in the LLM summary report. Defaults to 5.
        output : str, optional
            Output format. "text" (default) returns a markdown report for LLM evaluation; "dict" returns raw structured results.

        Returns
        -------
        str or dict
            Formatted English text report for LLM if `output="text"`, otherwise a raw results dictionary.

        Notes
        -----
        Heavy-Atom First Strategy (Z-contrast): When the structure contains elements with strong 
        scattering contrast (e.g., Ru vs O), set `element_types: ["Ru"]` during initial subgroup 
        screening. The heavy-atom framework determines the primary PDF profile and subgroup symmetry. 
        Light atom coordinates (e.g., O) should be refined in subsequent detailed stages to avoid 
        premature trapping in unphysical local minima.

        Examples
        --------
        >>> res = simulator.explore_subgroup_pathways(
        ...     lattice_parameters={"a": 0.3, "b": 0.3, "c": 0.0, "alpha": 0, "beta": 0, "gamma": 0},
        ...     atomic_coordinates={"site_indices": None, "element_types": ["Ru"], "max_change": 0.1},
        ...     wyckoff_positions={"site_indices": None, "element_types": None},
        ...     simulation_parameters={"scale": 0.1},
        ...     optimizer={"type": "bayes", "fit_min": 0.1, "fit_max": 15, "n_trials": 60, "sampler": "TPE"},
        ...     nest_optimizer={"type": "bayes", "fit_min": 0.1, "fit_max": 15, "n_trials": 300, "sampler": "CMA"}
        ... )
        """

        from .submodules.subgroup import explore_subgroup_pathways
        return explore_subgroup_pathways(self, 
                                         optimizer=optimizer,
                                         nest_optimizer=nest_optimizer,
                                         top_k=top_k,
                                         output=output,
                                         **kwargs)

    def refine(self, obs_x=None, obs_y=None, optimizer: dict = None, **kwargs):
        """
        Perform targeted or coupled refinement of structural and simulation parameters against PDF data.

        This function optimizes specified subsets of parameters (lattice constants, atomic positions, 
        particle morphology, ADPs, instrumental scale) using either local/least-squares or global/Bayesian 
        optimization algorithms.

        Parameters
        ----------
        obs_x : array_like, str, or OrochiBase, optional
            Observed experimental radial grid points (r in Å), or a file path to .gr/text,
            or a PDFget instance. If None, uses previously registered data via `set_obs_data()`.
        obs_y : array_like, optional
            Observed experimental G(r) values. Required if obs_x is an array.
        lattice_parameters : dict, optional
            Dictionary for lattice constant refinement limits.
            Keys:
                - a, b, c (float): Maximum allowed change for axis lengths (in Å). Set to 0.0 or omit to fix.
                - alpha, beta, gamma (float): Maximum allowed change for lattice angles (in degrees). Set to 0.0 or omit to fix.
            Example: {"a": 0.3, "b": 0.3, "c": 0.3, "alpha": 0.0, "beta": 0.0, "gamma": 0.0}
        atomic_coordinates : dict, optional
            Dictionary for fractional atomic coordinate refinement.
            Keys:
                - site_indices (list, optional): List of site indices to optimize. If None, all sites are considered.
                - element_types (list, optional): List of element symbols (e.g., ['O']) to restrict coordinate optimization. 
                    *RECOMMENDED STRATEGY*: Isolate light elements (e.g., ['O']) during later refinement stages.
                - max_change (float): Maximum allowed displacement in fractional units (e.g., 0.2).
            Example: {"site_indices": None, "element_types": ["O"], "max_change": 0.2}
        wyckoff_positions : dict, optional
            Dictionary for exploring Wyckoff positions.
            Keys:
                - site_indices (list, optional): List of site indices to explore. If None, all sites are considered.
                - element_types (list, optional): List of element symbols to restrict Wyckoff search.
            Example: {"site_indices": None, "element_types": None}
        particle_size : dict, optional
            Dictionary for nanoparticle morphology refinement.
            Keys:
                - min (float): Minimum nanoparticle diameter (in Å).
                - max (float): Maximum nanoparticle diameter (in Å).
            Example: {"min": 30.0, "max": 50.0}
        element_adps : dict, optional
            Dictionary for Atomic Displacement Parameters (ADPs / B-iso).
            Keys:
                - element_types (list, optional): List of element symbols (e.g., ['Ru', 'O']). If None, all elements are refined.
                - max_change (float): Maximum allowed change from current ADP values.
            Example: {"element_types": ['Ru', 'O'], "max_change": 0.01}
        simulation_parameters : dict, optional
            Dictionary for instrumental and simulation parameters.
            Keys:
                - scale (float): Max change for intensity scale factor. Set to 0.0 to fix.
                - delta1, delta2 (float): Max change for peak broadening. Set to 0.0 to fix.
                - qbroad, qdamp (float): Max change for instrumental factors. Set to 0.0 to fix.
            Example: {"scale": 0.1}
        optimizer : dict, optional
            Configuration dictionary for the optimization algorithm.
            Keys:
                - type (str): Framework ('bayes' or 'lls'). Defaults to 'bayes'.
                - fit_min (float): Minimum r-value (Å) for PDF fitting. Defaults to 0.1.
                - fit_max (float): Maximum r-value (Å) for PDF fitting. Defaults to 15.0.
                - n_trials (int): Number of optimization trials. Defaults to 300.
                - score_function (str, optional): Objective score calculation method.
                    * 'Rw' (default): Standard PDF Rw fitting error.
                    * 'rw_lj_penalty': Multiplicative Lennard-Jones/repulsive penalty (Rw * P_factor). 
                      P_factor is 1.0 in physically valid regions and scales up rapidly if atom distances 
                      (e.g., O-O < 2.2 Å) collapse. Recommended for basic light-element coordinate refinements.
                - sampler (str): Optuna sampling algorithm ('TPE' or 'CMA'). Defaults to 'TPE'. 
                                'CMA' is strongly recommended for atomic coordinate/polyhedral refinements.
                - study_name (str, optional): Optuna study/table name for RDB storage. Passing an existing 
                                          `study_name` allows resuming or appending additional trials 
                                          to a previous run (`load_if_exists=True`).
            Example: {
            "type": "bayes", "fit_min": 0.1, "fit_max": 15, "n_trials": 5000, "sampler": "CMA",
            "score_function": "rw_lj_penalty",
            "study_name": "oxygen_relaxation_cma_v1"
            }

        Returns
        -------
        dict
            Optimization result containing 'best_scores' and 'best_params'.

        Notes
        -----
        - **Z-Contrast & Stage Separation**: In systems with strong scattering contrast (e.g., Ru vs O), 
          NEVER refine light-atom coordinates simultaneously with macro-framework parameters (lattice, particle size). 
          Light-element scattering is weak and easily buried or distorted by macro-parameter adjustments.
        
        - **Step 1 (Macro Framework & Heavy Atoms)**: First refine lattice constants, particle size, scale, and heavy atom positions 
          while keeping light elements fixed (or omitted). Use a unique `study_name` (e.g., `"step1_macro_ru"`).
        
        - **Step 2 (Light Atom Relaxation)**: Fix the macro parameters (lattice, particle size) and refine ONLY light elements 
          (e.g., `element_types: ["O"]`). Use `sampler: "CMA"` with a high trial count (e.g., `n_trials: 5000`) and a dedicated 
          `study_name` (e.g., `"step2_oxygen_cma"`).
        
        - **Persistent Studies & Resuming / Appending Trials**: 
          By specifying `study_name` in `optimizer`, Optuna's search history is preserved in a database (`load_if_exists=True`). 
          If a refinement (especially CMA-ES in Step 2) did not fully converge due to an insufficient trial budget, 
          you can call `refine` again with the **SAME `study_name`** to seamlessly resume and execute additional trials.

        Examples
        --------
        >>> # Step 1: Refine macro parameters with a dedicated study_name
        >>> simulator.refine(
        ...     lattice_parameters={"a": 0.3, "b": 0.3, "c": 0.3, "alpha": 0, "beta": 0, "gamma": 0},
        ...     particle_size={"min": 30, "max": 50},
        ...     simulation_parameters={"scale": 0.1},
        ...     optimizer={
        ...         "type": "bayes", "fit_min": 0.1, "fit_max": 15, "n_trials": 300, "sampler": "CMA",
        ...         "study_name": "step1_macro_ru"
        ...     }
        ... )
        >>>
        >>> # Step 2: Refine ONLY light elements (Oxygen)
        >>> simulator.refine(
        ...     atomic_coordinates={"site_indices": None, "element_types": ["O"], "max_change": 0.2},
        ...     wyckoff_positions={"site_indices": None, "element_types": None},
        ...     simulation_parameters={"scale": 0.1},
        ...     optimizer={
        ...         "type": "bayes", "fit_min": 0.1, "fit_max": 15, "n_trials": 5000, "sampler": "CMA",
        ...         "study_name": "step2_oxygen_cma"
        ...     }
        ... )
        """

        from .submodules.refinement import refine

        if obs_x is not None:
            self.set_obs_data(obs_x, obs_y)

        if (self.obs_x is None or self.obs_y is None) and getattr(self, "_obs_module", None) is not None:
            if hasattr(self._obs_module, "__call__"):
                self._obs_module()

        if self.obs_x is None or self.obs_y is None:
            raise ValueError(
                "Observed experimental data is not set. Please call `simulator.set_obs_data(obs_x, obs_y)` "
                "or pass `obs_x` and `obs_y` directly to `refine()`."
            )
        if self._obs_x is None:
            self._obs_x = self.obs_x
            self._obs_y = self.obs_y

        return refine(self, optimizer=optimizer, **kwargs)

    def update_simulation_parameters(self, 
                                        scale: float = None, 
                                        delta1: float = None, 
                                        delta2: float = None, 
                                        qbroad: float = None, 
                                        qdamp: float = None):
        """
        Update instrumental and simulation-specific parameters for PDF calculation.

        Parameters
        ----------
        scale : float, optional
            Overall intensity scale factor. 
        delta1 : float, optional
            Peak broadening coefficient (1/r). 
        delta2 : float, optional
            Peak broadening coefficient (1/r^2). 
        qbroad : float, optional
            Instrumental resolution broadening factor. 
        qdamp : float, optional
            Instrumental resolution damping factor. 

        Returns
        -------
        str
            Log messages describing parameter update operations.

        Notes
        -----
        Only parameters with provided values (not None) will be updated. The default value is None, and no update is made.
        """
        updates = {
            "scale": scale,
            "delta1": delta1,
            "delta2": delta2,
            "qbroad": qbroad,
            "qdamp": qdamp
        }
        logs = ""

        for key, value in updates.items():
            if value is not None:
                if key in self:
                    self[key] = float(value)
                    logs += f"Parameter '{key}' updated to {value}.\n"
                else:
                    logs += f"Parameter key '{key}' not found in Simulator.\n"

        if logs:
            self._logger.info(logs)
        return logs