

def update_lattice_parameters(self, a, b, c, alpha, beta, gamma):
    self.phase.update_lattice_parameters(a, b, c, alpha, beta, gamma)

def update_atomic_coordinates(self, site_index, x, y, z):
    return self.phase.update_atomic_coordinates(site_index, x, y, z)

def update_particle_size(self, diameter):
    return self.phase.update_particle_size(diameter)

def update_element_adps(self, element, adp_value):
    return self.phase.update_element_adps(element, adp_value)

def make_subgroup_structure(self, hall_number: int, transformation:str):
    return self.phase.make_subgroup_structure(hall_number, transformation)

def make_supergroup_structure(self, hall_number: int, transformation:str):
    return self.phase.make_supergroup_structure(hall_number, transformation)

def make_supercell(self, a_axis, b_axis, c_axis):
    return self.phase.make_supercell(a_axis, b_axis, c_axis)

def make_higher_symmetry_structure(self, symprec):
    return self.phase.make_higher_symmetry_structure(symprec)

def update_site_adp(self, site_index, adp_value):
    return self.phase.update_site_adp(site_index, adp_value)
