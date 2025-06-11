from poisson_solver import poisson_solver_periodic, poisson_solver_periodic_safe
from new_mass_deposition import deposit_cic,deposit_ngp,deposit_tsc
from jeans_initial import create_particles,phi_plummer,plot_density_vs_r,plot_velocity_distribution,plot_velocity_dispersion_profile
from new_orbit_integrator import compute_phi, interpolate_to_particles,compute_grid_acceleration
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# === Simulation parameters ===
N = 256  # Grid size: N x N x N
box_size = 1.0
N_particles =  1000 #10000
center = N // 2
dt = 0.001
n_steps = 10000  #200
dp = 'tsc'  # 'ngp', 'cic', or 'tsc'
solver = 'isolated' # 'isolated', 'periodic ,'periodic_safe'(softening = 0 equal to periodic)
integrator = 'kdk'         # 'kdk' or 'dkd' or 'rk4' or 'hermite_individual'   or 'hermite_fixed'
self_force = False          # True or False
softening = 0.0001
velocity_scale = 2   #jeans equation, scale the velocity to get Q_J
a = 0.005

def compute_total_energy(positions, velocities, masses, N, box_size,dp,solver):
    """Compute total energy (kinetic + potential) of the system."""
    KE = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))

    #phi,weights = compute_phi(positions, masses, N, box_size, dp, solver, soft_len = softening)
    phi, weights= deposit_phi_tsc(positions, phi_plummer, N, box_size, boundary = "periodic")
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

def plot_potential(phi,title):
    phi = phi[:,:,center].T
    fig3, ax3 = plt.subplots()
    im = ax3.imshow(phi, extent=[0, box_size, 0, box_size], origin='lower',
                        vmin=np.min(phi), vmax=np.max(phi),cmap='viridis')
    cbar_potential = plt.colorbar(im, ax=ax3)    # add color bar
    cbar_potential.set_label("Gravitational Potential")
    #catter = ax3.scatter(positions[:,0], positions[:,1], s=1, color='white')

    #ax3.set_xlim(0, box_size)
    #ax3.set_ylim(0, box_size)
    ax3.set_xlim(box_size/4.0, 3.0*box_size/4)
    ax3.set_ylim(box_size/4.0, 3.0*box_size/4)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    title3 = ax3.set_title(f"Particles + Potential (from {title})")
    plt.tight_layout()
    plt.show()


# === initial setup ====
np.random.seed(42)
positions, velocities, masses = create_particles(N_particles,box_size,a = a, M = 1.0, mode="stable",G=1.0)

# check plummer distribution 
#"""
fig_init, ax_init = plt.subplots()
ax_init.scatter(positions[:, 0], positions[:, 1], s=5, alpha=0.6)
ax_init.set_aspect('equal')
ax_init.set_xlim(0, box_size)
ax_init.set_ylim(0, box_size)
ax_init.set_xlabel("x")
ax_init.set_ylabel("y")
ax_init.set_title("Initial Particle Positions (XY Plane)")
plt.tight_layout()
plt.show()
plot_density_vs_r(box_size,positions, masses,a = a, M=1.0)
plot_velocity_distribution(velocities)
plot_velocity_dispersion_profile(positions,velocities,a =a)
#"""

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

def deposit_phi_tsc(positions, phi, N, box_size, boundary):
    phi_grid = np.zeros((N, N, N))
    dx = box_size / N
    weights_list = []  # NEW: list of weights per particle

    def tsc_weights(d):
        w_m1 = 0.5 * (1.5 - d)**2
        w_0  = 0.75 - (d - 1.0)**2
        w_p1 = 0.5 * (d - 0.5)**2
        return np.array([w_m1, w_0, w_p1])

    for pos, phi in zip(positions, phi):
        xg = pos[0] / dx
        yg = pos[1] / dx
        zg = pos[2] / dx

        ix = int(np.floor(xg)) 
        iy = int(np.floor(yg)) 
        iz = int(np.floor(zg)) 

        dx1 = xg - ix
        dy1 = yg - iy
        dz1 = zg - iz

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
                        phi_grid[i, j, k] += phi * weight
                        particle_weights.append(((i, j, k), weight))
                    elif boundary == 'isolated':
                        if 0 <= i < N and 0 <= j < N and 0 <= k < N:
                            phi_grid[i, j, k] += phi * weight
                            particle_weights.append(((i, j, k), weight))

        weights_list.append(particle_weights)  # NEW

    return phi_grid, weights_list


phi_plummer = plummer_potential_from_positions(positions)  #in particle 
phi_grid, weighted= deposit_phi_tsc(positions, phi_plummer, N, box_size, boundary = "periodic")
print(phi_plummer)
plot_potential(phi_grid,"analytic")
acc_plummer = compute_grid_acceleration(phi_grid, N, box_size)
acc_plummer_par = interpolate_to_particles(acc_plummer, weighted)
accs_plummer = np.sqrt(acc_plummer_par[:,0]**2+acc_plummer_par[:,1]**2+acc_plummer_par[:,2]**2)


# === initial potential ===
phi_poisson, weighted = compute_phi(positions, masses, N, box_size, dp, solver, soft_len = softening)
plot_potential(phi_poisson,"poisson solver")

def interpolate_scalarfield(grid_field, weights_list):
    particle_values = []

    for weights in weights_list:
        value = 0.0
        for idx, w in weights:
            value += grid_field[idx] * w
        particle_values.append(value)

    return np.array(particle_values)

phi_par = interpolate_scalarfield(phi_poisson, weighted) #inparticle
acc_grid = compute_grid_acceleration(phi_poisson,N, box_size)
acc_par = interpolate_to_particles(acc_grid, weighted)
accs_par = np.sqrt(acc_par[:,0]**2+acc_par[:,1]**2+acc_par[:,2]**2)
print(phi_par)

# === error analyse ===
center_pos = np.array([box_size / 2] * 3)
r = np.linalg.norm(positions - center_pos, axis=1)

mask = r >= 0.01
r_valid = r[mask]
phi_true = phi_plummer[mask]
phi_numeric = phi_par[mask]
#acc_true = accs_plummer[mask]
#acc_numeric = accs_par[mask]

error = phi_numeric - phi_true #acc_numeric - acc_true #phi_numeric - phi_true
plt.figure(figsize=(6, 4))
plt.scatter(r_valid, error, s=2, alpha=0.5)
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("r")
plt.xscale('log')
plt.ylabel("Error: phi_numeric - phi_true")
plt.title("Potential Error vs Radius (excluding r < 0.01)")
plt.grid(True)
plt.tight_layout()
plt.show()

print("Error stats:")
print(f"Mean error     : {np.mean(error):.4e}")
print(f"Max error      : {np.max(error):.4e}")
print(f"Min error      : {np.min(error):.4e}")
print(f"Std deviation  : {np.std(error):.4e}")

