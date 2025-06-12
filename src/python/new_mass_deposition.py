import numpy as np

def deposit_ngp(positions, masses, N, box_size, boundary):
    rho = np.zeros((N, N, N))
    dx = box_size / N
    weights_list = []  # NEW: list of weights per particle

    for pos, m in zip(positions, masses):
        ix = int(np.floor(pos[0] / dx))
        iy = int(np.floor(pos[1] / dx))
        iz = int(np.floor(pos[2] / dx))
        if boundary == 'periodic':
            ix %= N
            iy %= N
            iz %= N
            rho[ix, iy, iz] += m / dx**3
        elif boundary == 'isolated':
            if 0 <= ix < N and 0 <= iy < N and 0 <= iz < N:
                rho[ix, iy, iz] += m / dx**3

        # NGP uses a single grid point per particle, no weights needed
        weights_list.append([((ix, iy, iz), 1)])  # NEW: single point with weight 1

    return rho, weights_list

def deposit_cic(positions, masses, N, box_size, boundary):
    rho = np.zeros((N, N, N))
    dx = box_size / N
    weights_list = []  # NEW: list of weights per particle

    for pos, m in zip(positions, masses):
        xg = pos[0] / dx -0.5
        yg = pos[1] / dx -0.5
        zg = pos[2] / dx -0.5

        ix = int(np.floor(xg))
        iy = int(np.floor(yg))
        iz = int(np.floor(zg))

        dx1 = xg - ix
        dy1 = yg - iy
        dz1 = zg - iz

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

        neighbors = [
            (ix    , iy    , iz    ),
            (ix+1  , iy    , iz    ),
            (ix    , iy+1  , iz    ),
            (ix    , iy    , iz+1  ),
            (ix+1  , iy+1  , iz    ),
            (ix+1  , iy    , iz+1  ),
            (ix    , iy+1  , iz+1  ),
            (ix+1  , iy+1  , iz+1  )
        ]

        particle_weights = []  # NEW

        for w, (i, j, k) in zip(weights, neighbors):
            if boundary == 'periodic':
                i %= N
                j %= N
                k %= N
                rho[i, j, k] += m * w / dx**3
                particle_weights.append(((i, j, k), w))  # NEW
            elif boundary == 'isolated':
                if 0 <= i < N and 0 <= j < N and 0 <= k < N:
                    rho[i, j, k] += m * w / dx**3
                    particle_weights.append(((i, j, k), w))  # NEW

        weights_list.append(particle_weights)  # NEW

    return rho, weights_list

def deposit_tsc(positions, masses, N, box_size, boundary):
    rho = np.zeros((N, N, N))
    dx = box_size / N
    weights_list = []  # NEW: list of weights per particle

    def tsc_weights(d):
        w_m1 = 0.5 * (1.5 - d)**2
        w_0  = 0.75 - (d - 1.0)**2
        w_p1 = 0.5 * (d - 0.5)**2
        return np.array([w_m1, w_0, w_p1])

    for pos, m in zip(positions, masses):
        xg = pos[0] / dx
        yg = pos[1] / dx
        zg = pos[2] / dx

        ix = int(np.floor(xg)) 
        iy = int(np.floor(yg)) 
        iz = int(np.floor(zg)) 

        dx1 = xg - ix -0.5
        dy1 = yg - iy -0.5
        dz1 = zg - iz -0.5

        wx = tsc_weights(dx1)
        wy = tsc_weights(dy1)
        wz = tsc_weights(dz1)

        particle_weights = []  # NEW

        for dx_idx in range(-1, 2):
            for dy_idx in range(-1, 2):
                for dz_idx in range(-1, 2):
                    i = ix + dx_idx
                    j = iy + dy_idx
                    k = iz + dz_idx
                    weight = wx[dx_idx+1] * wy[dy_idx+1] * wz[dz_idx+1]

                    if boundary == 'periodic':
                        i %= N
                        j %= N
                        k %= N
                        rho[i, j, k] += m * weight / dx**3
                        particle_weights.append(((i, j, k), weight))
                    elif boundary == 'isolated':
                        if 0 <= i < N and 0 <= j < N and 0 <= k < N:
                            rho[i, j, k] += m * weight / dx**3
                            particle_weights.append(((i, j, k), weight))

        weights_list.append(particle_weights)  # NEW

    return rho, weights_list