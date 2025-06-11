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


def deposit_tsc(positions, masses, N, box_size, boundary):
    """
    Triangular Shaped Cloud (TSC) deposition:
      positions: shape (M,3) array of particle coordinates
      masses:     shape (M,)   array of particle masses
      N:          grid size per dimension
      box_size:   physical size of the box
      boundary:   'periodic' or 'isolated'
    Returns
      rho:        (N,N,N) density grid
      weights_list: list of per-particle lists [((i,j,k), weight), ...]
    """
    rho = np.zeros((N, N, N), dtype=float)
    dx = box_size / N
    weights_list = []

    def tsc_w(r):
        """TSC weight for distance r (can be negative)."""
        ar = abs(r)
        if ar < 0.5:
            return 0.75 - ar**2
        elif ar < 1.5:
            return 0.5 * (1.5 - ar)**2
        else:
            return 0.0

    for pos, m in zip(positions, masses):
        # 1) 週期性映射
        if boundary == 'periodic':
            pos = np.mod(pos, box_size)

        # 2) 轉換到格點座標
        xg, yg, zg = pos / dx
        ix, iy, iz = int(np.floor(xg+0.5)), int(np.floor(yg+0.5)), int(np.floor(zg+0.5))

        particle_weights = []
        # 3) 對 27 個相鄰格點計算 TSC 權重
        for dx_idx in (-1, 0, 1):
            for dy_idx in (-1, 0, 1):
                for dz_idx in (-1, 0, 1):
                    i = ix + dx_idx
                    j = iy + dy_idx
                    k = iz + dz_idx

                    # 真實距離 r_x, r_y, r_z
                    wx = tsc_w(xg - (ix + dx_idx))
                    wy = tsc_w(yg - (iy + dy_idx))
                    wz = tsc_w(zg - (iz + dz_idx))
                    w = wx * wy * wz

                    # 邊界處理
                    if boundary == 'periodic':
                        i %= N; j %= N; k %= N
                    elif not (0 <= i < N and 0 <= j < N and 0 <= k < N):
                        continue

                    # 累加到 rho
                    rho[i, j, k] += m * w / dx**3
                    particle_weights.append(((i, j, k), w))

        weights_list.append(particle_weights)

    return rho, weights_list
