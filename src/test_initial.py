from poisson_solver import poisson_solver_periodic, poisson_solver_periodic_safe
from new_mass_deposition import deposit_cic,deposit_ngp,deposit_tsc
from jeans_initial import create_particles,phi_plummer,plot_density_vs_r,plot_velocity_distribution,plot_velocity_dispersion_profile
from new_orbit_integrator import compute_phi, interpolate_to_particles
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# === Simulation parameters ===
N = 64  # Grid size: N x N x N
box_size = 1.0
N_particles =  10000 #10000
center = N // 2
dt = 0.001
n_steps = 1000  #200
dp = 'ngp'  # 'ngp', 'cic', or 'tsc'
solver = 'periodic' # 'isolated', 'periodic ,'periodic_safe'(softening = 0 equal to periodic)
integrator = 'kdk'         # 'kdk' or 'dkd' or 'rk4' or 'hermite_individual'   or 'hermite_fixed'
self_force = True          # True or False
softening = 0.0
velocity_scale = 2   #jeans equation, scale the velocity to get Q_J
a = 0.01

def compute_total_energy(positions, velocities, masses, N, box_size,dp,solver):
    """Compute total energy (kinetic + potential) of the system."""
    KE = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))

    phi,weights = compute_phi(positions, masses, N, box_size, dp, solver, soft_len = softening)
    particle_values = []
    for weight in weights:
        phi_par = 0.0  # use scalar
        for idx, w in weight:
            phi_par += phi[idx] * w
        particle_values.append(phi_par)
    particle_values = np.array(particle_values)
    PE = 0.5*np.sum(masses*particle_values)
    return KE, PE


def compute_total_momentum(velocities, masses):
    """Compute momentum (vector)"""
    return np.sum(masses[:, None] * velocities, axis=0)

# === initial setup ====
np.random.seed(42)
positions, velocities, masses = create_particles(N_particles,box_size,a = a, M = 1.0, mode="stable",G=1.0)

# check plummer distribution 
fig_init, ax_init = plt.subplots()
ax_init.scatter(positions[:, 0], positions[:, 1], s=5, alpha=0.6)
ax_init.set_aspect('equal')
ax_init.set_xlim(0, box_size)
ax_init.set_ylim(0, box_size)
ax_init.set_xlabel("x")
ax_init.set_ylabel("y")
ax_init.set_title("Initial Particle Positions (XY Plane)")
plt.tight_layout()
#plt.show()
#plot_density_vs_r(box_size,positions, masses,a = a, M=1.0)
#plot_velocity_distribution(velocities)
#plot_velocity_dispersion_profile(positions,velocities,a =a)


# === check poisson solver
# === analytic potential ===
def plummer_potential_from_positions(positions, G=1.0, M=1.0, a= a, box_size=1.0):
    """
    Compute Plummer potential φ for each particle position, centered in the box.

    Parameters:
        positions: (N, 3) array of particle positions
        G: gravitational constant
        M: total mass
        a: Plummer scale length
        box_size: size of the simulation box (assumes cube from 0 to box_size)

    Returns:
        phi: (N,) array of potential values
    """
    center = np.array([box_size / 2] * 3)
    relative_positions = positions - center
    x = relative_positions[:,0]
    y = relative_positions[:,1]
    z = relative_positions[:,2]
    r = np.sqrt(x**2+y**2+z**2)
    phi = phi_plummer(r, a)
    return phi

phi = plummer_potential_from_positions(positions)  #in particle 
print(phi)

# === initial potential ===
phi_poisson, weighted = compute_phi(positions, masses, N, box_size, dp, solver, soft_len = softening)

def interpolate_scalarfield(grid_field, weights_list):
    particle_values = []

    for weights in weights_list:
        value = 0.0
        for idx, w in weights:
            value += grid_field[idx] * w
        particle_values.append(value)

    return np.array(particle_values)

phi_par = interpolate_scalarfield(phi_poisson, weighted) #inparticle
print(phi_par)