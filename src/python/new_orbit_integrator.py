import numpy as np
from poisson_solver import poisson_solver_periodic,poisson_solver_isolated
from new_mass_deposition import deposit_cic,deposit_ngp,deposit_tsc


def compute_grid_acceleration( phi, N, box_size):

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

  
def nbody_compute_acceleration(positions, masses, N, box_size):
    particle_values = []
    N_particles =len(masses)
    for i in range(N_particles):
        acc = np.zeros(3)
        for j in range(0, N_particles):
            if i!=j:
                dx = positions[i] - positions[j]
                r = np.sqrt(dx[0]**2+dx[1]**2+dx[2]**2 + box_size/N**2 )
                acc[0] -= 1.0 *  masses[j]*dx[0] / r**3
                acc[1] -= 1.0 *  masses[j]*dx[1] / r**3
                acc[2] -= 1.0 *  masses[j]*dx[2] / r**3

        particle_values.append(acc)
    acc_direct = np.array(particle_values)

    #calculate grid phi
    phi = np.zeros((N,N,N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N_particles):
                    r = np.sqrt((positions[l,0]-i/N)**2+
                                (positions[l,1]-j/N)**2+
                                (positions[l,2]-k/N)**2+ box_size/N**2 )
                    phi[i,j,k] -= 1.0 *  masses[l] / r
    return acc_direct,phi


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




#---------------------------------------------------------------------------------------------------------------#
#------------------------------------------------Hermite--------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#


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

def hermite_step_fixed_pm(particles, N, box_size, dp, solver, subtract_self=False, G=1.0):
    
    dt = particles[0]['dt']  
    positions = np.array([p['r'] for p in particles])
    masses = np.array([p['m'] for p in particles])

    # 1. Predict
    for p in particles:
        p['r_pred'] = p['r'] + p['v'] * dt + 0.5 * p['a'] * dt**2 + (1/6) * p['j'] * dt**3
        p['v_pred'] = p['v'] + p['a'] * dt + 0.5 * p['j'] * dt**2

    # 2.  (PM) acc
    pred_positions = np.array([p['r_pred'] for p in particles])
    a_new, phi = compute_acceleration(pred_positions, masses, N, box_size, dp, solver, subtract_self)

    # 3. jerk = (a_new - a_old)/dt
    for i, p in enumerate(particles):
        p['j_new'] = (a_new[i] - p['a']) / dt

    # 4. Correct
    for i, p in enumerate(particles):
        r_pred = p['r_pred']
        v_pred = p['v_pred']
        a_old = p['a']
        j_old = p['j']
        a_new_i = a_new[i]
        j_new_i = p['j_new']

        p['r'] = r_pred + (1/24) * (a_new_i - a_old) * dt**2 + (1/120) * (j_new_i - j_old) * dt**3
        p['v'] = v_pred + (1/6) * (a_new_i - a_old) * dt + (1/24) * (j_new_i - j_old) * dt**2
        p['a'] = a_new_i
        p['j'] = j_new_i
        p['t'] += dt

    positions = np.array([p['r'] for p in particles])
    velocities = np.array([p['v'] for p in particles])

    return positions, velocities, phi

