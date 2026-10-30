.. image:: images/orochi_icon.png
   :align: center
   :width: 80px
   :target: https://github.com/Tominaka-Laboratory/orochi.pdfsim
   :alt: Orochi Logo

=============
orochi.pdfsim
=============

.. |title| replace:: orochi.pdfsim
.. _title: https://Tominaka-Laboratory.github.io/orochi.pdfsim

.. |Icon| image:: images/orochi_icon.png
   :target: https://github.com/Tominaka-Laboratory/orochi.pdfsim
   :height: 100px

|PyPi| |Forge| |PythonVersion| |ChemRxiv| |PR|

|CI| |Codecov| |Black| |Tracking|

.. |ChemRxiv| image:: https://img.shields.io/badge/ChemRxiv-10.26434%2Fchemrxiv.15010802%2Fv2-blue.svg
        :target: https://doi.org/10.26434/chemrxiv.15010802/v2/v2

.. |Black| image:: https://img.shields.io/badge/code_style-black-black
        :target: https://github.com/psf/black

.. |CI| image:: https://github.com/Tominaka-Laboratory/orochi.pdfsim/actions/workflows/tests.yml/badge.svg
        :target: https://github.com/Tominaka-Laboratory/orochi.pdfsim/actions/workflows/tests.yml

.. |Codecov| image:: https://codecov.io/gh/Tominaka-Laboratory/orochi.pdfsim/branch/main/graph/badge.svg
        :target: https://codecov.io/gh/Tominaka-Laboratory/orochi.pdfsim

.. |Forge| image:: https://img.shields.io/conda/vn/conda-forge/orochi.pdfsim
        :target: https://anaconda.org/conda-forge/orochi.pdfsim

.. |PR| image:: https://img.shields.io/badge/PR-Welcome-29ab47ff

.. |PyPi| image:: https://img.shields.io/pypi/v/orochi.pdfsim
        :target: https://pypi.org/project/orochi.pdfsim/

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/orochi.pdfsim
        :target: https://pypi.org/project/orochi.pdfsim/

.. |Tracking| image:: https://img.shields.io/badge/issue_tracking-github-blue
        :target: https://github.com/Tominaka-Laboratory/orochi.pdfsim/issues



Overview
========

The ``orochi.pdfsim`` package is a Python module dedicated to the calculation
of Pair Distribution Functions (PDFs), denoted as *G(r)*. It interfaces with
``orochi.phase`` to obtain structural information and computes the PDF
considering periodic boundary conditions.

The core calculations are optimized for speed using **Numpy** and **Numba**.
While primarily designed for CPU-based computation, GPU support for
large-scale calculations is under development to further enhance performance.

A key characteristic of ``orochi.pdfsim`` is its ability to incorporate the
scattering angle dependence (Q-dependence) of atomic scattering factors.
This feature is particularly advantageous for accurately simulating PDFs of
materials containing pairs of atoms with significantly different atomic numbers.

Key Features
============

* **Pair Distribution Function (PDF) Simulation**:
    Calculates *G(r)* from structural models.
* **Integration with ``orochi.phase``**:
    Seamlessly retrieves atomic coordinates, occupancies, displacement parameters,
    and lattice information.
* **Periodic Boundary Conditions**:
    Correctly handles periodic structures to accurately compute pair distances
    across unit cell boundaries.
* **High-Performance Core**:
    Utilizes Numpy and Numba for efficient computation on CPUs.
* **GPU Acceleration (Upcoming)**:
    GPU support is currently being updated to enable faster processing of
    large-scale simulations.
* **Q-dependent Scattering Factors**:
    Allows for the inclusion of the Q-dependence of atomic scattering factors,
    providing more accurate PDF simulations, especially for systems with
    heavy and light element combinations.

Quickstart: Simulating G(r)
=============================

.. code-block:: python

    from orochi.phase import Phase
    from orochi.pdfsim import Simulator
    import matplotlib.pyplot as plt

    # 1. Load crystallographic structure
    phase = Phase()
    phase.read_file("CeO2.cif")
    phase["shape_type"] = "analytical"   # Bulk crystal representation

    # 2. Configure Simulator and link Phase
    sim = Simulator()
    sim.phase = phase
    sim["rambda"] = 0.18033              # X-ray wavelength in Angstrom
    sim["qmin"] = 1.0
    sim["qmax"] = 25.0
    sim["rmin"] = 1.5
    sim["rmax"] = 30.0
    sim["qdamp"] = 0.028                 # Instrumental resolution damping
    sim["qdepend"] = True                # Exact Q-dependent form factors via DCT

    # 3. Execution
    # Standard call utilizes internal caching (only recalculates modified dependencies):
    sim()

    # Alternatively, use run(force=True) to guarantee full recalculation:
    # sim.run(force=True)

    # 4. Access calculated PDF arrays
    r = sim.r
    gr = sim.Gr

    plt.plot(r, gr, label="Calculated G(r)")
    plt.xlabel("r (Å)")
    plt.ylabel("G(r) (Å⁻²)")
    plt.legend()
    plt.show()

Execution and Caching Lifecycle
===============================

``orochi.pdfsim`` inherits from ``orochi.base.core.OrochiBase`` and incorporates dependency-aware method wrapping:

* **Initial Execution**: On the first call (``_state == "INIT"``), the entire pipeline executes: atomic coordinate expansion, shape envelope convolution, atomic form factors, pair histograms *g_ij(r)*, and *G(r)* synthesis. Upon completion, state transitions to ``"DONE"``.
* **Intelligent Incremental Recalculation (``sim()``)**: During optimization (e.g. refining scale factor or resolution damping *Q_damp*), expensive pair distance calculations (``calc_rij``) are **automatically skipped** because geometric parameters have not changed.
* **Forced Recalculation (``sim.run(force=True)``)**: When parameters are modified outside standard dictionaries or when absolute state reset is desired, passing ``force=True`` resets internal states to ``"INIT"``, enforcing complete end-to-end recomputation.

Citation
--------

If you use orochi.pdfsim in a scientific publication, please cite this package as:

    S. Tominaka, *Orochi: A Physics-Grounded Python Package for Pair Distribution Function Analysis of Crystalline and Low-Dimensional Materials*, ChemRxiv (2026). DOI: 10.26434/chemrxiv.15010802/v2
    https://doi.org/10.26434/chemrxiv.15010802/v2

Installation
------------

The preferred method is to install using ``pip``:

.. code-block:: bash

    pip install orochi.pdfsim

To confirm that the installation was successful, type:

.. code-block:: bash

    python -c "import orochi.pdfsim; print('Orochi PDFsim imported successfully')"

If you prefer to install from source:

.. code-block:: bash

    git clone https://github.com/Tominaka-Laboratory/orochi.pdfsim.git
    cd orochi.pdfsim
    pip install -e .

Getting Started
---------------

You may consult our `online documentation <https://Tominaka-Laboratory.github.io/orochi.pdfsim>`_ for tutorials and API references.

Support and Contribute
----------------------

The orochi.pdfsim project welcomes your expertise and enthusiasm!

If you see a bug or want to request a feature, please `report it as an issue <https://github.com/Tominaka-Laboratory/orochi.pdfsim/issues>`_ and/or `submit a fix as a PR <https://github.com/Tominaka-Laboratory/orochi.pdfsim/pulls>`_. 

Feel free to fork the project and contribute. To install orochi.pdfsim
in a development mode, with its sources being directly used by Python
rather than copied to a package directory, use the following in the root
directory ::

        pip install -e .

To ensure code quality and to prevent accidental commits into the default branch, please set up the use of our pre-commit
hooks.

Improvements and fixes are always appreciated.

Before contributing, please read our `Code of Conduct <https://github.com/Tominaka-Laboratory/orochi.pdfsim/blob/main/CODE_OF_CONDUCT.rst>`_.



License
-------

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**. See the `LICENSE.rst` file for details.

This copyleft license guarantees that Orochi remains freely accessible, open, and transparent to the entire scientific community, and that all derivative works must likewise remain open source.

Citation
--------

If you use this package in academic research, please cite:

.. code-block:: bibtex

    @article{tominaka2026orochi,
      title   = {Orochi: A Physics-Grounded Python Package for Pair Distribution Function Analysis of Crystalline and Low-Dimensional Materials},
      author  = {Tominaka, Satoshi},
      journal = {ChemRxiv},
      year    = {2026},
      doi     = {10.26434/chemrxiv.15010802/v2},
      url     = {https://doi.org/10.26434/chemrxiv.15010802/v2}
    }

Contact
-------

For more information on orochi.pdfsim please visit the project `web-page <https://Tominaka-Laboratory.github.io/>`_ or email Dr. Satoshi Tominaka at tominaka.satoshi@nims.go.jp.
