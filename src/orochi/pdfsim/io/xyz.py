def export_all_atoms_xyz(self, savefile=""):
    if self["shape_type"] == "analytical":
        data = np.concatenate([self.symbols.reshape(-1, 1), self.xyz], axis=1)
        data = data.astype(str)
        header = str(int(self.symbols.size)) + "\n"
    else:
        data = np.concatenate(
            [self.symbols_all.reshape(-1, 1), self.xyz_all], axis=1
        )
        data = data.astype(str)
        header = str(int(self.symbols_all.size)) + "\n"

    np.savetxt(savefile, data, fmt="%s", header=header, comments="")
