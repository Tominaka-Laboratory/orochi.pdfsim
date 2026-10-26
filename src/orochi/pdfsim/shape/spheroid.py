# spheroidal_matrix is 3x3 matrix
def spheroidal_boundary(self, xyz, spheroidal_matrix, center):
    xyz_shape = np.append(xyz.shape[:-1], 9)
    xyz = xyz.reshape(-1, 3) - center

    xyz1 = np.tile(xyz, (1, 3)).reshape(xyz_shape)
    xyz2 = np.tile(xyz.flatten(), (3, 1)).T.reshape(xyz_shape)

    matrix = xyz1 * xyz2 * spheroidal_matrix.flatten()
    inspheroid = np.where(np.sum(matrix, axis=-1) <= 1.0)

    return inspheroid
