import numpy as np
from poisson_solver import poisson_solver_periodic,poisson_solver_isolated
from new_mass_deposition import deposit_cic,deposit_ngp,deposit_tsc

def compute_grid_acceleration(phi, N, box_size):
    """ compute acceleration at grids"""
    dx = box_size / N
    # Compute gradients on the grid (finite differences)
    grad_phi = np.gradient(-phi, dx, edge_order=2)  # returns [grad_phi_x, grad_phi_y, grad_phi_z]

    return grad_phi

def interpolate_to_particles(grid_field, weights_list):
    """
    Interpolate grid field (scalar or vector) back to particle positions using precomputed weights.

    Parameters
    ----------
    grid_field : ndarray
        Grid values, shape (N, N, N) or (3, N, N, N) for vector field (e.g., acceleration).
    weights_list : list of list of (index tuple, weight)
        Each entry corresponds to a particle's interpolation stencil.

    Returns
    -------
    particle_values : ndarray
        Interpolated values at particle positions. Shape:
        - (N_particles,) for scalar field
        - (N_particles, 3) for vector field
    """
    particle_values = []

    for weights in weights_list:
        acc = np.zeros(3)
        for idx, w in weights:
            acc[0] += grid_field[0][idx] * w
            acc[1] += grid_field[1][idx] * w
            acc[2] += grid_field[2][idx] * w
        particle_values.append(acc)

    return np.array(particle_values)

def compute_phi(positions, masses, N, box_size, dp, solver,G_k):
    """
    Solve Poisson equation from positions and masses via:
    1. Mass deposition
    2. Solve Poisson equation
    """

    boundary = 'periodic' if solver.startswith('periodic') else solver

    # Step 1: Mass deposition
    if dp == "ngp":
        rho, weights = deposit_ngp(positions, masses, N, box_size, boundary)
        #rho, weights = example_omp.deposit_ngp(positions, masses, N, box_size, boundary)
    elif dp == "cic":
        rho, weights = deposit_cic(positions, masses, N, box_size, boundary)
        #rho, weights = example_omp.deposit_cic(positions, masses, N, box_size, boundary)
    elif dp == "tsc":
        rho, weights = deposit_tsc(positions, masses, N, box_size, boundary)
        #rho, weights = example_omp.deposit_tsc(positions, masses, N, box_size, boundary)
    else:
        raise ValueError("Unknown deposition method")
    
    #print("rho min/max:", np.min(rho), np.max(rho))

    # Step 2: Solve Poisson equation
    if solver == "periodic":
        phi = poisson_solver_periodic(rho, box_size, G=1.0)
    elif solver == "isolated":
        phi = poisson_solver_isolated(rho, G_k,N, box_size, G=1.0)
    else:
        raise ValueError("Unknown boundary condition")

    return phi,weights

def compute_acceleration(positions, masses, N, box_size, dp, solver,G_k):
    """
    Computes particle accelerations from positions and masses via:
    1. Compute potential
    2. Compute grid acceleration
    3. Interpolate acceleration back to particles
    """
    #Step 1: Compute potential
    phi,weights = compute_phi(positions, masses, N, box_size, dp, solver,G_k)
    # Step 2: Compute grid acceleration
    acc_grid = compute_grid_acceleration(phi, N, box_size)
    
    # Step 3: Interpolate to particles
    acc_particles = interpolate_to_particles(acc_grid, weights)

    return acc_particles, phi

def kdk_step(positions, velocities, masses, dt, N, box_size, dp, solver, G_k):
    """
    Perform one KDK (Kick-Drift-Kick) integration step with full acceleration computation.
    """
    boundary = solver
    # First Kick (half step)
    acc,phi = compute_acceleration(positions, masses, N, box_size, dp, solver, G_k)
    velocities += 0.5 * dt * acc

    # Drift (full step)
    positions += dt * velocities
    #positions %= box_size  # periodic boundary condition

    if boundary == "periodic":
        positions %= box_size
    elif boundary == "isolated":
        mask = np.all((positions >= 0) & (positions < box_size), axis=1)
        positions = positions[mask]
        velocities = velocities[mask]
        masses = masses[mask]

    # Second Kick (half step)
    acc,phi = compute_acceleration(positions, masses, N, box_size, dp, solver,G_k)
    velocities += 0.5 * dt * acc

    return positions, velocities,masses, phi

def dkd_step(positions, velocities, masses, dt, N, box_size, dp,solver,G_k):
    """
    Perform one DKD (Drift-Kick-Drift) integration step with acceleration computation.
    """
    boundary = 'periodic' if solver.startswith('periodic') else solver
    # First Drift (half step)
    positions += 0.5 * dt * velocities
    #positions %= box_size  # periodic boundary condition
    if boundary == "periodic":
        positions %= box_size
    elif boundary == "isolated":
        mask = np.all((positions >= 0) & (positions < box_size), axis=1)
        positions = positions[mask]
        velocities = velocities[mask]
        masses = masses[mask]

    # Compute acceleration at updated positions
    acc,phi = compute_acceleration(positions, masses, N, box_size, dp,solver,G_k)

    # Kick (full step)
    velocities += dt * acc

    # Second Drift (half step)
    positions += 0.5 * dt * velocities
    #positions %= box_size  # periodic boundary condition
    if boundary == "periodic":
        positions %= box_size
    elif boundary == "isolated":
        mask = np.all((positions >= 0) & (positions < box_size), axis=1)
        positions = positions[mask]
        velocities = velocities[mask]
        masses = masses[mask]

    return positions, velocities, masses, phi

def rk4_step(positions, velocities, masses, dt, N, box_size,dp, solver,G_k,G = 1.0):
    """
    4th-order RK for collision-less gravity on a periodic mesh.
    """
    boundary = 'periodic' if solver.startswith('periodic') else solver

    # RK4 stages
    a1 ,phi = compute_acceleration(positions, masses, N, box_size, dp,solver,G_k)
    k1x = dt * velocities
    k1v = dt * a1

    a2,phi = compute_acceleration(positions + 0.5*k1x, masses, N, box_size, dp,solver,G_k)
    k2x = dt * (velocities + 0.5*k1v)
    k2v = dt * a2

    a3,phi = compute_acceleration(positions + 0.5*k2x, masses, N, box_size, dp,solver, G_k)
    k3x = dt * (velocities + 0.5*k2v)
    k3v = dt * a3

    a4,phi = compute_acceleration(positions + k3x, masses, N, box_size, dp,solver,G_k)
    k4x = dt * (velocities + k3v)
    k4v = dt * a4

    new_pos = (positions +
               (k1x + 2*k2x + 2*k3x + k4x)/6.0)
    new_vel = velocities + \
              (k1v + 2*k2v + 2*k3v + k4v)/6.0
    
    if boundary == "periodic":
        new_pos %= box_size
    elif boundary == "isolated":
        mask = np.all((new_pos >= 0) & (new_pos < box_size), axis=1)
        new_pos = new_pos[mask]
        new_vel = new_vel[mask]
        masses = masses[mask]
    
    new_phi,w = compute_phi(new_pos, masses, N, box_size, dp, solver,G_k)
    return new_pos, new_vel, masses, new_phi

