import numpy as np

par = {
    # "phase": {"value": None},
    "rmin": {"value": [0.0], "min": [0.0], "max": [100000.0]},
    "rmax": {"value": [20.0], "min": [0.001], "max": [100000.0]},
    "rstep": {"value": [0.01], "min": [0.001], "max": [10.0]},
    "qmin": {"value": [0.5], "min": [0.001], "max": [100.0]},
    "qmax": {"value": [20.0], "min": [0.001], "max": [100.0]},
    "qdamp": {"value": [0.0], "min": [0.0], "max": [1.0], "fit": False},
    "qbroad": {"value": [0.0], "min": [0.0], "max": [1.0], "fit": False},
    "scale": {"value": [1.0], "min": [0.0001], "max": [100.0], "fit": False},
    "rambda": {"value": [1.0], "min": [0.1], "max": [2.0], "fit": False},
    "energy": {"value": [8.0], "min": [3], "max": [120], "fit": False},
    "rmax_extension": {"value": [4.0], "min": [0.0], "max": [10.0]},
    "gaus_width": {"value": [5.0], "min": [0.1], "max": [10.0]},
    "pad": {"value": [5], "min": [1], "max": [10]},
    "peak_data_point":{"value":100},
    # "number_scf_function": {"value": [100], "min": [1], "max": [200]},
    "delta1":{"value":0.0, "min":0.0, "max":5, "fit": False},
    "delta2":{"value":0.0, "min":0.0, "max":5, "fit": False},
    "sratio": {
        "value": np.array([[1.]]),
        "pair": np.array([["X", "X"]]),
        "range": np.array([[0, 0]]),
        "min": np.array([[0.1]]),
        "max": np.array([[2.]]),
        "fit": [[False, False]],
    },
    "qdepend": {"value": True},  # or constant, epf
    "density_type": {"value": "analytical", "member":["analytical", "simulation"]},  # simulation or analytical
    "simulation_type": {"value": "pdf"},  # or epf
    "processor_type": {"value": "cpu"},  # or gpu
    "N_indep": {"value":0}
    # "shape_type": {"value": "analytical", "member":["analytical", "sphere"]},
    # "sampling_probability": {"value": 1.0},
    # "partial_epf_origin": {"value": [0, 0, 0]},
    # "mesh_size": {"value": [0.01]},
    # "gr_deconv": {
    #     "value": np.empty((0), float),
    #     "min": np.empty((0), float),
    #     "max": np.empty((0), float),
    # },
    # "gr_gauss": {"value": [0.01], "min": [0.0], "max": [5.0]},
}

par_key_info = {
    # "phase": {"value": [object, None, 1]},
    "rmin": {
        "value": [float, None, 1],
        "min": [float],
        "max": [float],
    },
    "rmax": {
        "value": [float, None, 1],
        "min": [float],
        "max": [float],
    },
    "rstep": {
        "value": [float, None, 1],
        "min": [float],
        "max": [float],
    },
    "qmin": {
        "value": [float, None, 1],
        "min": [float],
        "max": [float],
    },
    "qmax": {
        "value": [float, None, 1],
        "min": [float],
        "max": [float],
    },
    "qdamp": {
        "value": [float, None, 1],
        "min": [float],
        "max": [float],
    },
    "qbroad": {
        "value": [float, None, 1],
        "min": [float],
        "max": [float],
    },
    "scale": {
        "value": [float, None, 1],
        "min": [float],
        "max": [float],
    },
    "rambda": {
        "value": [float, None, 1],
        "min": [float],
        "max": [float],
    },
    "energy": {
        "value": [float, None, 1],
        "min": [float],
        "max": [float],
    },
    "rmax_extension": {
        "value": [float, None, 1],
        "min": [float],
        "max": [float],
    },
    "gaus_width": {
        "value": [float, None, 1],
        "min": [float],
        "max": [float],
    },
    "pad": {"value": [int, None, 1], "min": [int], "max": [int]},
    "peak_data_point": {"value": [int, None, 1]},
    # "number_scf_function": {
    #     "value": [int, None, 1],
    #     "min": [int],
    #     "max": [int],
    # },
    "delta1": {
        "value": [float, None, 1],
        "min": [float],
        "max": [float],
    },
    "delta2": {
        "value": [float, None, 1],
        "min": [float],
        "max": [float],
    },
    "sratio": {
        "value": [float, None, (-1, 1)],
        "range": [float, None, (-1, 2)],
        "pair": [str, None, (-1, 2)],
        "min": [float, None, (-1, 2)],
        "max": [float, None, (-1, 2)],
    },
    "qdepend": {"value": [bool, None, 1]},
    "density_type": {"value": [str, None, 1]},
    "simulation_type": {"value": [str, None, 1]},
    "processor_type": {"value": [str, None, 1]},
    "N_indep": {"value": [int, None, 1]},
    "shape_type": {"value": [str, None, 1]},
    # "sampling_probability": {
    #     "value": [float, None, 1],
    #     "min": [float],
    #     "max": [float],
    # },
    # "partial_epf_origin": {"value": [float, None, (3)]},
    # "optimize": {"value": [str, None, 1]},
    # "mesh_size": {"value": [float, None, 1]},
}
