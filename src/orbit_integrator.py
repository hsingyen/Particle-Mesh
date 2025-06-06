import numpy as np
from itertools import product
from poisson_solver import poisson_solver_periodic_safe

def compute_acceleration(positions, phi, N, box_size):
    """
    Compute gravitational acceleration -grad(phi) at particle positions.
    Use nearest-grid interpolation for now (simple NGP).

    Parameters
    ----------
    positions : ndarray
        (N_particles, 3) array of particle positions.
    phi : ndarray
        (N, N, N) gravitational potential on the grid.
    N : int
        Grid size.
    box_size : float
        Physical size of the box.

    Returns
    -------
    accelerations : ndarray
        (N_particles, 3) array of particle accelerations.
    """
    dx = box_size / N
    accelerations = np.zeros_like(positions)

    # Compute gradients on the grid (finite differences)
    grad_phi = np.gradient(phi, dx, edge_order=2)  # returns [grad_phi_x, grad_phi_y, grad_phi_z]

    for i, pos in enumerate(positions):
        # Find nearest grid point
        ix = int(np.round(pos[0] / dx)) % N
        iy = int(np.round(pos[1] / dx)) % N
        iz = int(np.round(pos[2] / dx)) % N

        # Acceleration is -gradient of potential
        ax = -grad_phi[0][ix, iy, iz]
        ay = -grad_phi[1][ix, iy, iz]
        az = -grad_phi[2][ix, iy, iz]

        accelerations[i] = np.array([ax, ay, az])

    return accelerations

def kdk_step(positions, velocities, masses, dt, phi, N, box_size):
    """
    Perform one KDK (Kick-Drift-Kick) integration step.

    Parameters
    ----------
    positions : ndarray
        (N_particles, 3) particle positions
    velocities : ndarray
        (N_particles, 3) particle velocities
    masses : ndarray
        (N_particles,) particle masses (not used here yet, but keep for later)
    dt : float
        Time step
    phi : ndarray
        (N, N, N) gravitational potential
    N : int
        Grid size
    box_size : float
        Physical size of box

    Returns
    -------
    positions, velocities : ndarray
        Updated particle positions and velocities
    """
    # First Kick (half step)
    acc = compute_acceleration(positions, phi, N, box_size)
    velocities += 0.5 * dt * acc

    # Drift (full step)
    positions += dt * velocities
    positions %= box_size  # periodic boundary condition

    # Recompute potential here if needed (optional for simplicity now)
    # Assuming phi is static during dt

    # Second Kick (half step)
    acc = compute_acceleration(positions, phi, N, box_size)
    velocities += 0.5 * dt * acc

    return positions, velocities


def dkd_step(positions, velocities, masses, dt, phi, N, box_size):
    """
    Perform one DKD (Drift-Kick-Drift) integration step.

    Parameters
    ----------
    positions : ndarray
        (N_particles, 3) particle positions
    velocities : ndarray
        (N_particles, 3) particle velocities
    masses : ndarray
        (N_particles,) particle masses (not used here yet)
    dt : float
        Time step
    phi : ndarray
        (N, N, N) gravitational potential
    N : int
        Grid size
    box_size : float
        Physical size of box

    Returns
    -------
    positions, velocities : ndarray
        Updated particle positions and velocities
    """
    dx = box_size / N

    # First Drift (half step)
    positions += 0.5 * dt * velocities
    positions %= box_size  # Apply periodic boundary condition

    # Compute acceleration at mid positions
    acc = compute_acceleration(positions, phi, N, box_size)

    # Full Kick
    velocities += dt * acc

    # Second Drift (half step)
    positions += 0.5 * dt * velocities
    positions %= box_size  # Apply periodic boundary condition again

    return positions, velocities


import numpy as np
from poisson_solver import poisson_solver_periodic_safe


# ---------------------------------------------------------------------
#  fast CIC force interpolation (all-NumPy, no take_along_axis)
# ---------------------------------------------------------------------
def cic_acceleration(positions, phi, box_size):
    N   = phi.shape[0]
    dx  = box_size / N
    inv = 1.0 / dx

    # central-difference gradient on the grid
    gx = (np.roll(phi, -1, 0) - np.roll(phi, 1, 0)) * (0.5 * inv)
    gy = (np.roll(phi, -1, 1) - np.roll(phi, 1, 1)) * (0.5 * inv)
    gz = (np.roll(phi, -1, 2) - np.roll(phi, 1, 2)) * (0.5 * inv)

    gpos = positions * inv
    i0   = np.floor(gpos).astype(int)      # cell origin
    d    = gpos - i0                       # fractional offset

    # weights for the 8 vertices of the cell
    w = np.empty((positions.shape[0], 8))
    w[:,0] = (1-d[:,0])*(1-d[:,1])*(1-d[:,2])
    w[:,1] =    d[:,0] *(1-d[:,1])*(1-d[:,2])
    w[:,2] = (1-d[:,0])*   d[:,1] *(1-d[:,2])
    w[:,3] = (1-d[:,0])*(1-d[:,1])*   d[:,2]
    w[:,4] =    d[:,0] *   d[:,1] *(1-d[:,2])
    w[:,5] =    d[:,0] *(1-d[:,1])*   d[:,2]
    w[:,6] = (1-d[:,0])*   d[:,1] *   d[:,2]
    w[:,7] =    d[:,0] *   d[:,1] *   d[:,2]

    shifts = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],
                       [1,1,0],[1,0,1],[0,1,1],[1,1,1]])
    ip = (i0[:,None,:] + shifts[None,:,:]) % N        # (Np,8,3)

    # gather –∇φ on the 8 vertices (shape → (Np,8))
    ax = -gx[ip[:,:,0], ip[:,:,1], ip[:,:,2]]
    ay = -gy[ip[:,:,0], ip[:,:,1], ip[:,:,2]]
    az = -gz[ip[:,:,0], ip[:,:,1], ip[:,:,2]]

    acc = np.stack((np.sum(w*ax,1),
                    np.sum(w*ay,1),
                    np.sum(w*az,1)), axis=1)
    return acc


# ---------------------------------------------------------------------
#  your integrator – only this function is new/changed
# ---------------------------------------------------------------------
def rk4_step(positions, velocities, masses,
             dt, N, box_size,
             deposit_func,
             soft_len = 0.0,
             G        = 1.0):
    """
    4th-order RK for collision-less gravity on a periodic mesh.
    """
    def acceleration(pos):
        rho = deposit_func(pos % box_size, masses, N, box_size)
        phi = poisson_solver_periodic_safe(rho, box_size,
                                           G=G, soft_len=soft_len)
        return cic_acceleration(pos, phi, box_size)

    # RK4 stages
    a1  = acceleration(positions)
    k1x = dt * velocities
    k1v = dt * a1

    a2  = acceleration(positions + 0.5*k1x)
    k2x = dt * (velocities + 0.5*k1v)
    k2v = dt * a2

    a3  = acceleration(positions + 0.5*k2x)
    k3x = dt * (velocities + 0.5*k2v)
    k3v = dt * a3

    a4  = acceleration(positions + k3x)
    k4x = dt * (velocities + k3v)
    k4v = dt * a4

    new_pos = (positions +
               (k1x + 2*k2x + 2*k3x + k4x)/6.0) % box_size
    new_vel = velocities + \
              (k1v + 2*k2v + 2*k3v + k4v)/6.0
    return new_pos, new_vel











#Hermite

def predict_all(particles, t_target):
    """Return predicted r and v for all particles at future time t_target."""
    predicted = []
    for p in particles:
        dt = t_target - p['t']
        r_pred = p['r'] + p['v'] * dt + 0.5 * p['a'] * dt**2 + (1/6) * p['j'] * dt**3
        v_pred = p['v'] + p['a'] * dt + 0.5 * p['j'] * dt**2
        predicted.append({'r': r_pred, 'v': v_pred, 'm': p['m']})
    return predicted


def compute_force_and_jerk(i, predicted, G=1.0, eps=1e-10):
    """Compute acceleration and jerk for particle i using predicted states."""
    r_i = predicted[i]['r']
    v_i = predicted[i]['v']
    a = np.zeros(3)
    j = np.zeros(3)

    for jdx, pj in enumerate(predicted):
        if jdx == i:
            continue
        r_j = pj['r']
        v_j = pj['v']
        m_j = pj['m']

        dr = r_j - r_i
        dv = v_j - v_i
        r2 = np.dot(dr, dr) + eps
        r3 = r2 * np.sqrt(r2)
        r5 = r2 * r3

        a += G * m_j * dr / r3
        j += G * m_j * (dv / r3 - 3 * np.dot(dr, dv) * dr / r5)

    return a, j

def correct(p, r_pred, v_pred, a_new, j_new, t_target):
    """Correct position and velocity of one particle using Hermite corrector."""
    dt = t_target - p['t']
    a_old = p['a']
    j_old = p['j']

    # Correct
    p['r'] = r_pred + (1/24) * (a_new - a_old) * dt**2 + (1/120) * (j_new - j_old) * dt**3
    p['v'] = v_pred + (1/6) * (a_new - a_old) * dt + (1/24) * (j_new - j_old) * dt**2
    p['a'] = a_new
    p['j'] = j_new
    p['t'] = t_target


def update_dt(p, eta=0.01):
    """Update time step using Aarseth criterion."""
    a = np.linalg.norm(p['a'])
    j = np.linalg.norm(p['j'])
    if j == 0:
        return p['dt']
    return eta * np.sqrt(a / j)

def hermite_individual_step(particles, G=1.0):
    """
    Advance the particle with the smallest (t + dt) using Hermite integrator.
    """
    N = len(particles)

    # 1. Select particle with soonest update time
    times = [p['t'] + p['dt'] for p in particles]
    i = np.argmin(times)
    t_target = times[i]

    # 2. Predict all particles to t_target
    predicted = predict_all(particles, t_target)

    # 3. Compute new a and j
    a_new, j_new = compute_force_and_jerk(i, predicted, G)

    # 4. Correct r, v for particle i
    correct(particles[i], predicted[i]['r'], predicted[i]['v'], a_new, j_new, t_target)

    # 5. Recalculate time step
    particles[i]['dt'] = update_dt(particles[i])

    return particles[i]['t']



def hermite_step_fixed(particles, G=1.0):
    """
    Advance all particles one step using Hermite 4th-order predictor-corrector integrator.
    This version uses a shared fixed time step for simplicity.

    Parameters:
    -----------
    particles : list of dict
        Each dict must have keys:
        'r': position (3D array), 
        'v': velocity (3D array),
        'a': acceleration (3D array),
        'j': jerk (3D array),
        'm': mass (float)
    G : float
        Gravitational constant (default = 1)

    Returns:
    --------
    updated_particles : list of dict
        Updated particle list with r, v, a, j values after one Hermite step
    """
    N = len(particles)
    dt = particles[0]['dt']  # assuming shared dt for now

    # Predict
    for p in particles:
        r = p['r']
        v = p['v']
        a = p['a']
        j = p['j']

        p['r_pred'] = r + v * dt + 0.5 * a * dt**2 + (1/6) * j * dt**3
        p['v_pred'] = v + a * dt + 0.5 * j * dt**2

    # Recompute a' and j' using predicted r and v
    for i in range(N):
        r_i = particles[i]['r_pred']
        v_i = particles[i]['v_pred']
        a_new = np.zeros(3)
        j_new = np.zeros(3)
        for j in range(N):
            if i == j:
                continue
            r_j = particles[j]['r_pred']
            v_j = particles[j]['v_pred']
            m_j = particles[j]['m']
            dr = r_j - r_i
            dv = v_j - v_i
            r2 = np.dot(dr, dr) + 1e-10  # softening to avoid zero division
            r3 = r2 * np.sqrt(r2)
            r5 = r2 * r3

            a_new += G * m_j * dr / r3
            j_new += G * m_j * (dv / r3 - 3 * np.dot(dr, dv) * dr / r5)

        particles[i]['a_new'] = a_new
        particles[i]['j_new'] = j_new

    # Correct
    for p in particles:
        r_pred = p['r_pred']
        v_pred = p['v_pred']
        a_old = p['a']
        j_old = p['j']
        a_new = p['a_new']
        j_new = p['j_new']

        p['r'] = r_pred + (1/24) * (a_new - a_old) * dt**2 + (1/120) * (j_new - j_old) * dt**3
        p['v'] = v_pred + (1/6) * (a_new - a_old) * dt + (1/24) * (j_new - j_old) * dt**2
        p['a'] = a_new
        p['j'] = j_new

    return particles