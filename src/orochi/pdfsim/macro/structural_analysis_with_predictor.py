def calc_coordinate_distances_to_standard_structure(xyz_std):
    def scoring(phase):
        xyz = phase.par["xyz"]
        xyz_pbc = (
            xyz[:, :, np.newaxis]
            + np.array([0, 1, -1])[np.newaxis, np.newaxis, :]
        )
        dist_vector = xyz_pbc - xyz_std[:, :, np.newaxis]

        res = np.argmin(np.abs(dist_vector), axis=-1)
        shortest_vector = np.take_along_axis(
            dist_vector, np.expand_dims(res, axis=-1), axis=-1
        ).reshape(xyz.shape)
        dist = np.linalg.norm(shortest_vector, axis=-1)
        return dist

    return scoring
