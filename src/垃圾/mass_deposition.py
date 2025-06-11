import numpy as np

def deposit_ngp(positions, masses, N, box_size):
    """
    Deposit particle masses onto a 3D grid using Nearest Grid Point (NGP) scheme.

    Parameters
    ----------
    positions : ndarray
        Array of shape (N_particles, 3) with particle positions [x, y, z].
    masses : ndarray
        Array of shape (N_particles,) with particle masses.
    N : int
        Grid size (assume cubic grid: N x N x N).
    box_size : float
        Size of the simulation box.

    Returns
    -------
    rho : ndarray
        3D mass density field.
    """
    rho = np.zeros((N, N, N))

    dx = box_size / N

    for pos, m in zip(positions, masses):
        ix = int(np.round(pos[0] / dx)) % N
        iy = int(np.round(pos[1] / dx)) % N
        iz = int(np.round(pos[2] / dx)) % N
        rho[ix, iy, iz] += m / dx**3  # convert mass to density

    return rho


def deposit_cic(positions, masses, N, box_size):
    """
    Deposit particle masses onto a 3D grid using Cloud-In-Cell (CIC) scheme.

    Parameters
    ----------
    positions : ndarray
        Array of shape (N_particles, 3) with particle positions [x, y, z].
    masses : ndarray
        Array of shape (N_particles,) with particle masses.
    N : int
        Grid size (assume cubic grid: N x N x N).
    box_size : float
        Size of the simulation box.

    Returns
    -------
    rho : ndarray
        3D mass density field.
    """
    rho = np.zeros((N, N, N))
    dx = box_size / N

    for pos, m in zip(positions, masses):
        # Find base grid index
        xg = pos[0] / dx
        yg = pos[1] / dx
        zg = pos[2] / dx

        ix = int(np.floor(xg)) % N
        iy = int(np.floor(yg)) % N
        iz = int(np.floor(zg)) % N

        # Distances to lower grid point
        dx1 = xg - ix
        dy1 = yg - iy
        dz1 = zg - iz

        # CIC weights
        weights = np.array([
            (1-dx1)*(1-dy1)*(1-dz1),
            dx1    *(1-dy1)*(1-dz1),
            (1-dx1)*dy1    *(1-dz1),
            (1-dx1)*(1-dy1)*dz1,
            dx1    *dy1    *(1-dz1),
            dx1    *(1-dy1)*dz1,
            (1-dx1)*dy1    *dz1,
            dx1    *dy1    *dz1
        ])

        # Neighboring grid points
        neighbors = [
            (ix    , iy    , iz    ),
            ((ix+1)%N, iy    , iz    ),
            (ix    , (iy+1)%N, iz    ),
            (ix    , iy    , (iz+1)%N),
            ((ix+1)%N, (iy+1)%N, iz    ),
            ((ix+1)%N, iy    , (iz+1)%N),
            (ix    , (iy+1)%N, (iz+1)%N),
            ((ix+1)%N, (iy+1)%N, (iz+1)%N)
        ]

        for w, (i,j,k) in zip(weights, neighbors):
            rho[i,j,k] += m * w / dx**3

    return rho


def deposit_tsc(positions, masses, N, box_size):
    """
    Deposit particle masses onto a 3D grid using Triangular Shaped Cloud (TSC) scheme.

    Parameters
    ----------
    positions : ndarray
        Array of shape (N_particles, 3) with particle positions [x, y, z].
    masses : ndarray
        Array of shape (N_particles,) with particle masses.
    N : int
        Grid size (assume cubic grid: N x N x N).
    box_size : float
        Size of the simulation box.

    Returns
    -------
    rho : ndarray
        3D mass density field.
    """
    rho = np.zeros((N, N, N))
    dx = box_size / N

    for pos, m in zip(positions, masses):
        # Normalize position to grid units
        xg = pos[0] / dx
        yg = pos[1] / dx
        zg = pos[2] / dx

        ix = int(np.floor(xg)) % N
        iy = int(np.floor(yg)) % N
        iz = int(np.floor(zg)) % N

        # Distances from particle to grid point
        dx1 = xg - ix
        dy1 = yg - iy
        dz1 = zg - iz

        # Compute weights in 1D for x, y, z
        def tsc_weights(d):
            w_m1 = 0.5 * (1.5 - d)**2
            w_0  = 0.75 - (d - 1.0)**2
            w_p1 = 0.5 * (d - 0.5)**2
            return np.array([w_m1, w_0, w_p1])

        wx = tsc_weights(dx1)
        wy = tsc_weights(dy1)
        wz = tsc_weights(dz1)

        # Neighboring grid points
        for dx_idx in range(-1, 2):
            for dy_idx in range(-1, 2):
                for dz_idx in range(-1, 2):
                    i = (ix + dx_idx) % N
                    j = (iy + dy_idx) % N
                    k = (iz + dz_idx) % N
                    weight = wx[dx_idx+1] * wy[dy_idx+1] * wz[dz_idx+1]
                    rho[i, j, k] += m * weight / dx**3

    return rho