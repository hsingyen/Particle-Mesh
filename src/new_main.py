import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from poisson_solver import poisson_solver_periodic, poisson_solver_periodic_safe, poisson_solver_isolated
from new_mass_deposition import deposit_ngp, deposit_cic, deposit_tsc
from new_orbit_integrator import kdk_step, dkd_step, hermite_step_fixed, hermite_individual_step,rk4_step,hermite_step_fixed_pm
from utils import Timer  # optional
from mpl_toolkits.mplot3d import Axes3D

# === Simulation parameters ===
N = 64  # Grid size: N x N x N
box_size = 1.0
N_particles =  1000 #10000
center = N // 2
dt = 0.001
n_steps = 1000  #200
dp = 'ngp'  # 'ngp', 'cic', or 'tsc'
solver = 'periodic_safe' # 'isolated', 'periodic ,'periodic_safe'
integrator = 'kdk'         # 'kdk' or 'dkd' or 'rk4' or 'hermite_individual'   or 'hermite_fixed'

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
    radius = 0.05 * box_size  # Sphere radius

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

# periodic boundary energy
def compute_total_energy(positions, velocities, masses, phi, N, box_size):
    """Compute total energy (kinetic + potential) of the system."""
    KE = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))

    dx = box_size / N
    potential = np.zeros(len(positions))
    for i, pos in enumerate(positions):
        ix = int(np.round(pos[0] / dx)) % N
        iy = int(np.round(pos[1] / dx)) % N
        iz = int(np.round(pos[2] / dx)) % N
        potential[i] = phi[ix, iy, iz]

    PE = 0.5 * np.sum(masses * potential)
    return KE, PE

def compute_total_momentum(velocities, masses):
    """Compute momentum (vector)"""
    return np.sum(masses[:, None] * velocities, axis=0)

# === Main Simulation ===
def main():
    np.random.seed(42)
    #positions, velocities, masses = create_random_particles(N_particles, box_size)
    # test self-gravity collapse
    positions, velocities, masses = create_random_center_particles(N_particles, box_size)
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
    saved_frames = set()

    for step in range(n_steps):
        # Orbit integration
        ## change input parameter
        if integrator == 'kdk':
            positions, velocities,phi = kdk_step(positions, velocities, masses, dt, N, box_size, dp, solver, subtract_self=True)
        elif integrator == 'dkd':
            positions, velocities,phi = dkd_step(positions, velocities, masses, dt, N, box_size, dp, solver)
        elif integrator == 'rk4':
            positions, velocities,phi = rk4_step(positions, velocities, masses, dt, N, box_size, dp, solver)
        elif integrator == 'hermite_fixed':
            positions, velocities,phi = hermite_step_fixed_pm(positions, velocities, masses, dt, N, box_size, dp, solver)
        
        # add hermite scheme


        #Energy
        KE, PE = compute_total_energy(positions, velocities, masses, phi, N, box_size)
        energies.append([KE, PE])
        #Momentum
        current_momentum = compute_total_momentum(velocities, masses)
        delta_P = current_momentum - initial_momentum
        momentum_errors.append(np.linalg.norm(delta_P))
        momentum_errors_xyz.append(np.abs(delta_P))

        #print(f"Step {step}")
        #print("  KE =", KE)
        #print("  PE =", PE)
        #print("  ΔP =", delta_P)
        #print("  Positions sample:", positions[0])
        #print("  Velocities sample:", velocities[0])
        #print()


        # Save frames every 2 steps
        if step % 2 == 0:
            frames.append(phi[:,:,center].copy())
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
    #ani_combined.save("particle_simulation.gif", writer=PillowWriter(fps=5))

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

if __name__ == "__main__":
    main()

