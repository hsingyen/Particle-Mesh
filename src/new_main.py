import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
#from new_mass_deposition import deposit_ngp, deposit_cic, deposit_tsc
from new_orbit_integrator import kdk_step, dkd_step,rk4_step
from new_orbit_integrator import compute_phi
from utils import Timer  # optional
from mpl_toolkits.mplot3d import Axes3D
from jeans_initial import create_particles

# === Simulation parameters ===
N = 128  # Grid size: N x N x N
box_size = 1.0
N_particles =  10000 #10000
center = N // 2
dt = 0.001
n_steps = 1000  #200
dp = 'ngp'  # 'ngp', 'cic', or 'tsc'
solver = 'periodic' # 'isolated', 'periodic ,'periodic_safe'(softening = 0 equal to periodic)
integrator = 'kdk'         # 'kdk' or 'dkd' or 'rk4' or 'hermite_individual'   or 'hermite_fixed'
self_force = True          # True or False
softening = 0.01 
velocity_scale = 5   #jeans equation, scale the velocity to get Q_J 
a = 0.005

# === Utility functions ===
def create_point_mass(N):
    """Create a single point mass density field."""
    rho = np.zeros((N, N, N))
    rho[N//2, N//2, N//2] = 1.0
    return rho

def create_random_particles(N_particles, box_size):
    """Generate random particle positions and masses."""
    positions = np.random.rand(N_particles, 3) * box_size
    velocities = np.zeros((N_particles, 3))
    masses = np.ones(N_particles) * (1.0 / N_particles)
    return positions, velocities, masses

# test self-gravity collapse
def create_random_center_particles(N_particles, box_size):
    """Generate random particle positions inside a sphere centered in the box."""
    center = np.array([box_size / 2] * 3)
    radius = 0.1 * box_size  # Sphere radius

    positions = []
    while len(positions) < N_particles:
        point = np.random.uniform(-radius, radius, size=3)
        if np.linalg.norm(point) <= radius:
            positions.append(center + point)

    positions = np.array(positions)
    velocities = np.zeros((N_particles, 3))
    masses = np.ones(N_particles) * (1.0 / N_particles)
    return positions, velocities, masses

def plot_slice(phi, box_size, title):
    """Plot mid-plane slice of potential."""
    plt.imshow(phi[:,:,center], extent=[0, box_size, 0, box_size], origin='lower')
    plt.colorbar(label="Potential Φ")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


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


def compute_energies_direct(x, v, m, G=1.0, softening=1e-5):
    #can be removed
    N = len(m)
    KE = 0.5 * np.sum(m * np.sum(v**2, axis=1))
    PE = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            r2 = np.dot(dx, dx) + softening**2
            PE -= G * m[i] * m[j] / np.sqrt(r2)
    return KE, PE, KE + PE

def compute_central_density(positions, radius=0.05, center=np.array([0.5, 0.5, 0.5])):
    d = np.linalg.norm(positions - center, axis=1)
    within = d < radius
    return np.sum(within) / ((4/3) * np.pi * radius**3)


# === Main Simulation ===
def main():
    np.random.seed(42)
    #positions, velocities, masses = create_random_particles(N_particles, box_size)
    # test self-gravity collapse
    #positions, velocities, masses = create_random_center_particles(N_particles, box_size)
    #jeans equation
    positions, velocities, masses = create_particles(N_particles, box_size, a = a , M =1.0, mode='stable',r_max = 5, G = 1.0)

    # Manually scale the velocities
    #velocities *= velocity_scale

    # Compare direct N-body energy with PM energy
    '''
    KE_direct, PE_direct, E_direct = compute_energies_direct(positions, velocities, masses, G=1.0, softening=softening)
    print(f"[Initial Direct Energy] KE = {KE_direct:.4e}, PE = {PE_direct:.4e}, Total = {E_direct:.4e}")

    KE_pm, PE_pm = compute_total_energy(positions, velocities, masses, N, box_size, dp, solver)
    E_pm = KE_pm + PE_pm
    print(f"[Initial PM Energy]     KE = {KE_pm:.4e}, PE = {PE_pm:.4e}, Total = {E_pm:.4e}")

    print(f"[Energy Differences]")
    print(f"ΔKE = {KE_pm - KE_direct:.4e}")
    print(f"ΔPE = {PE_pm - PE_direct:.4e}")
    print(f"ΔTotal = {E_pm - E_direct:.4e}")

    print(f"[Relative Errors]")
    print(f"ΔKE / KE_direct = {(KE_pm - KE_direct)/abs(KE_direct):.4e}")
    print(f"ΔPE / PE_direct = {(PE_pm - PE_direct)/abs(PE_direct):.4e}")
    print(f"ΔTotal / E_direct = {(E_pm - E_direct)/abs(E_direct):.4e}")
    '''

    # Initial Jeans Q_J calculation
    KE, PE = compute_total_energy(positions, velocities, masses, N, box_size, dp, solver)
    Q_J = 2 * KE / abs(PE)
    print(f"Initial Jeans Q_J = {Q_J:.2f}")

    fig_init, ax_init = plt.subplots()
    ax_init.scatter(positions[:, 0], positions[:, 1], s=5, alpha=0.6)
    circle = plt.Circle((box_size / 2, box_size / 2), 0.05 * box_size, color='r', fill=False, linestyle='--')
    ax_init.add_patch(circle)
    ax_init.set_aspect('equal')
    ax_init.set_xlim(0, box_size)
    ax_init.set_ylim(0, box_size)
    ax_init.set_xlabel("x")
    ax_init.set_ylabel("y")
    ax_init.set_title("Initial Particle Positions (XY Plane)")
    plt.tight_layout()
    plt.show()
    #print("Position range:", positions.min(), positions.max())
    #print("Velocity sample:", velocities[:3])
    #print("Total mass:", masses.sum())



    initial_momentum = compute_total_momentum(velocities, masses)
    momentum_errors = []         # total error ||ΔP||
    momentum_errors_xyz = []     # per-axis error [|ΔPx|, |ΔPy|, |ΔPz|]

    frames = []           # potential field frames
    particle_frames = []  # particle position frames
    energies = []
    KE_initial, PE_initial = compute_total_energy(positions, velocities, masses, N, box_size,dp,solver)
    energies.append([KE_initial,PE_initial])
    
    central_densities = []
    mean_radii = []

    saved_frames = set()

            

    for step in range(n_steps):
        # Orbit integration
        #print(step)
        ## change input parameter
        if integrator == 'kdk':
            positions, velocities, masses, phi = kdk_step(positions, velocities, masses, dt, N, box_size, dp, solver, subtract_self=self_force,soft_len=softening)
        elif integrator == 'dkd':
            positions, velocities,masses, phi = dkd_step(positions, velocities, masses, dt, N, box_size, dp, solver, subtract_self=self_force,soft_len=softening)
        elif integrator == 'rk4':
            positions, velocities,masses, phi = rk4_step(positions, velocities, masses, dt, N, box_size, dp, solver, subtract_self=self_force,soft_len=softening)


        # add hermite scheme


        #Energy
        #KE, PE = compute_total_energy(positions, velocities, masses, phi, N, box_size)
        KE, PE = compute_total_energy(positions, velocities, masses, N, box_size,dp,solver)
        energies.append([KE, PE])
        #Momentum
        current_momentum = compute_total_momentum(velocities, masses)
        delta_P = current_momentum - initial_momentum
        momentum_errors.append(np.linalg.norm(delta_P))
        momentum_errors_xyz.append(np.abs(delta_P))


        central_density = compute_central_density(positions)
        central_densities.append(central_density)
        mean_r = np.mean(np.linalg.norm(positions - np.array([0.5, 0.5, 0.5]), axis=1))
        mean_radii.append(mean_r)

        #print(f"Step {step}")
        #print("  KE =", KE)
        #print("  PE =", PE)
        #print("  ΔP =", delta_P)
        #print("  Positions sample:", positions[0])
        #print("  Velocities sample:", velocities[0])
        #print()


        # Save frames every 2 steps
        if step % 2 == 0:
            frames.append(phi[:,:,center].T.copy())
            particle_frames.append(positions.copy())
    
    # --- Combined Potential + Particles Animation ---
    fig3, ax3 = plt.subplots()
    im = ax3.imshow(frames[0], extent=[0, box_size, 0, box_size], origin='lower',
                    vmin=np.min(frames[0]), vmax=np.max(frames[0]), cmap='viridis')
    cbar_potential = plt.colorbar(im, ax=ax3)    # add color bar
    cbar_potential.set_label("Gravitational Potential")
    scatter = ax3.scatter(particle_frames[0][:,0], particle_frames[0][:,1], s=1, color='white')

    ax3.set_xlim(0, box_size)
    ax3.set_ylim(0, box_size)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    title3 = ax3.set_title(f"Particles + Potential at Frame 0")
    plt.tight_layout()

    def animate_combined(i):
        im.set_data(frames[i])
        scatter.set_offsets(particle_frames[i])
        title3.set_text(f"Particles + Potential at Frame {i}")
        return im, scatter, title3
    
    ani_combined = animation.FuncAnimation(fig3, animate_combined, frames=len(frames), interval=200, blit=False)
    plt.show()

    # --- Energy Conservation Plot ---
    energies = np.array(energies)
    KEs = energies[:,0]
    PEs = energies[:,1]
    TEs = KEs + PEs

    plt.figure()
    plt.plot(KEs, label="Kinetic Energy")
    plt.plot(PEs, label="Potential Energy")
    plt.plot(TEs, label="Total Energy")
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.title("Energy Conservation Test")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(TEs-TEs[0], label="Delta Total Energy")
    plt.xlabel("Step")
    plt.ylabel("Energy diff")
    plt.title("Energy Conservation Test")
    plt.legend()
    plt.grid()
    plt.show()

    # ---Momentum conservation plot ---
    momentum_errors_xyz = np.array(momentum_errors_xyz)  # shape: (n_steps, 3)

    plt.figure()
    plt.plot(momentum_errors_xyz[:, 0], label='|ΔP_x|')
    plt.plot(momentum_errors_xyz[:, 1], label='|ΔP_y|')
    plt.plot(momentum_errors_xyz[:, 2], label='|ΔP_z|')
    plt.xlabel("Step")
    plt.ylabel("Momentum Error")
    plt.title("Momentum Conservation Error per Component")
    plt.legend()
    plt.grid()
    plt.show()

    '''
    # --- Central Density Evolution ---
    plt.plot(mean_radii)
    plt.xlabel("Step")
    plt.ylabel("Mean Radius")
    plt.title("System Expansion")
    plt.figure()
    plt.plot(central_densities)
    plt.xlabel("Step")
    plt.ylabel("Central Density")
    plt.title("Central Density Evolution")
    plt.grid()
    plt.show()
    '''

if __name__ == "__main__":
    main()
