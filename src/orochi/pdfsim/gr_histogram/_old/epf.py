# This is for future work


def read_phase_for_epf(self):
    self.read_phase_lattice()
    self.read_phase_shape()
    self.calc_translational_vector()


def calc_gr_epf(self):
    r, gr = self.calc_gr_histogram_for_epf()
    gr *= (self["qmax"] * self["pad"]) / np.pi

    density = self.calc_density(gr, r)
    gr = self.convolution_with_q(gr, r, self["qmax"])
    self.gr = intp.interp1d(r, gr, fill_value="extrapolate")(self.r)
    self.density = intp.interp1d(r, density, fill_value="extrapolate")(self.r)


def calc_gr_histogram_for_epf(self):
    xyz = self["phase"]["map", "xyz"].reshape(-1, 3)
    xyz = np.dot(self.g, xyz.T).T

    map = self["phase"]["map", "value"]
    map = map.flatten()
    self.Np = np.sum(map)

    rstep = self.r[1] - self.r[0]
    r_extension = self.num_p_peak / self["qmax"]
    r = np.arange(0, self.r[-1] + rstep + r_extension, rstep)

    tvs = self.translational_vectors.reshape(-1, 3)

    if self["sampling_probability"] != 1.0:
        probability = self["sampling_probability"]
        rand = np.random.rand(map.size)
        use = np.where((rand < probability) | (map > np.amax(map) / 100.0))
        xyz = xyz[use]
        ratio = np.sum(map[use]) / np.sum(map)
        map = map[use]
        # self.par.logger.info("sampling probability: " + str(probability))

    if self["simulation_type"] == "epf":
        if self["processor_type"] == "cpu":
            from orochi.pdfsim.calculator.epf_cpu import calc_epf_cpu

            gr = calc_epf_cpu(map, xyz, tvs, r)
            return r, gr / ratio**2.0

        else:
            from orochi.pdfsim.calculator.epf_gpu import calc_epf_gpu

            gr = calc_epf_gpu(map, xyz, tvs, r)
            return r, gr / ratio**2.0

    elif self["simulation_type"] == "partial_epf":
        from orochi.pdfsim.calculator.epf_partial import calc_partial_epf

        origin = np.dot(self.g, np.array(self["partial_epf_origin"]))
        return r, calc_partial_epf(map, xyz, tvs, r, origin)
