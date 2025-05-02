import numpy as np

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