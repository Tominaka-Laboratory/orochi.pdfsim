import numpy as np


def save_Gr(out_path):
    def saving(module, name) -> None:
        r = module.r
        Gr = module.Gr
        data = np.concatenate([r.reshape(-1, 1), Gr.reshape(-1, 1)], axis=-1)
        np.savetxt(out_path + "/" + name, data, fmt="%1.4e")

    return saving
