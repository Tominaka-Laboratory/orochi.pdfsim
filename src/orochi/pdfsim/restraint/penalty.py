import numpy as np


def get_restraint_penalty(self):  # for optimization
    self.window_restraint = 0  # 0:true, >0:false
    self.potential_restraint = 0.0  # Lennard-Jones potential

    if self["window"][0][0] == self["window"][0][1]:  # restraint is off
        return self.window_restraint, self.potential_restraint

    window = self["window"]
    pair = self["window", "pair"]
    wrange = self["window", "range"]
    scale = self["window", "scale"]

    for id, (atom1, atom2) in enumerate(pair):
        ra = np.where((self.ai == atom1) | (self.ni == atom1), 1, 0)
        rb = np.where((self.aj == atom2) | (self.nj == atom2), 1, 0)
        rc = np.where((self.ai == atom2) | (self.ni == atom2), 1, 0)
        rd = np.where((self.aj == atom1) | (self.nj == atom1), 1, 0)
        r = ra * rb + rc * rd

        r_pair = np.where(r != 0)
        rij = self.rij[r_pair]

        # used ranges
        r_pair = np.where(((rij > wrange[id][0]) & (rij < wrange[id][1])))
        rij = rij[r_pair]

        # window restraint
        sum_violent_lower_limit = rij[rij < window[id][0]] - window[id][0]
        sum_violent_lower_limit = np.sum(np.abs(sum_violent_lower_limit))

        sum_violent_upper_limit = rij[rij > window[id][1]] - window[id][1]
        sum_violent_upper_limit = np.sum(np.abs(sum_violent_upper_limit))

        self.window_restraint += (
            sum_violent_lower_limit + sum_violent_upper_limit
        ) * scale[id]

        # bond_length restraint in the LJ type function
        bond_length = window[id][1] - window[id][0]
        ratio = bond_length / rij
        self.potential_restraint += (
            np.average(ratio**12.0 - 2.0 * ratio**6.0) * scale
        )

    self.potential_restraint /= np.shape(pair)[0]
    return self.window_restraint, self.potential_restraint[0]
