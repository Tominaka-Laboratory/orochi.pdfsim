"""not maintained yet

"""


def refine_correlation(simulator):
    simulator.par["correlation", "value"] = [
        [0.8, 1.0, 1.0],
        [0.8, 1.0, 1.0],
        [0.8, 1.0, 1.0],
    ]
    simulator.par["correlation", "pair"] = [
        ["Pt", "Pt"],
        ["Pt", "O"],
        ["Pt", "Pt"],
    ]
    simulator.par["correlation", "range"] = [[0, 6], [0, 2.5], [0, 30]]
    simulator["correlation", "fit"] = [
        [True, False, False],
        [True, False, False],
        [False, True, True],
    ]
    simulator.par["correlation", "min"] = [
        [0.3, 0.5, 0.0],
        [0.2, 0.5, 0.0],
        [0.5, 0.01, 0.0],
    ]
    simulator.par["correlation", "max"] = [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 2.0],
    ]
    return simulator
