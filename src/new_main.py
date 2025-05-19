import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from poisson_solver import poisson_solver_periodic
from new_mass_deposition import deposit_ngp, deposit_cic, deposit_tsc
from new_orbit_integrator import kdk_step, dkd_step, hermite_step_fixed, hermite_individual_step,rk4_step
from utils import Timer  # optional
from mpl_toolkits.mplot3d import Axes3D

# === Simulation parameters ===
N = 32  # Grid size: N x N x N
box_size = 1.0
N_particles =  100 #10000
center = N // 2
dt = 0.0001
n_steps = 100  #200
dp = 'cic'  # 'ngp', 'cic', or 'tsc'
solver = 'periodic' # 'isolated', 'periodic ,'periodic_safe'
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
    positions, velocities, masses = create_random_particles(N_particles, box_size)

    #print("Position range:", positions.min(), positions.max())
    #print("Velocity sample:", velocities[:3])
    #print("Total mass:", masses.sum())


    initial_momentum = compute_total_momentum(velocities, masses)
    momentum_errors = []         # total error ||ΔP||
    momentum_errors_xyz = []     # per-axis error [|ΔPx|, |ΔPy|, |ΔPz|]

    frames = []           # potential field frames
    particle_frames = []  # particle position frames
    energies = []
    density_frames = []

    for step in range(n_steps):
        # Orbit integration
        ## change input parameter
        if integrator == 'kdk':
            positions, velocities,phi = kdk_step(positions, velocities, masses, dt, N, box_size, dp, solver, subtract_self=True)
        elif integrator == 'dkd':
            positions, velocities,phi = dkd_step(positions, velocities, masses, dt, N, box_size, dp, solver, subtract_self=True)
        elif integrator == 'rk4':
            positions, velocities,phi = rk4_step(positions, velocities, masses, dt, N, box_size, dp, solver, subtract_self=True)
        
        # add hermite scheme


        #Energy
        KE, PE = compute_total_energy(positions, velocities, masses, phi, N, box_size)
        energies.append([KE, PE])
        #Momentum
        current_momentum = compute_total_momentum(velocities, masses)
        delta_P = current_momentum - initial_momentum
        momentum_errors.append(np.linalg.norm(delta_P))
        momentum_errors_xyz.append(np.abs(delta_P))

        print(f"Step {step}")
        print("  KE =", KE)
        print("  PE =", PE)
        print("  Total E =", KE + PE)
        print("  ΔP =", delta_P)
        print("  Positions sample:", positions[0])
        print("  Velocities sample:", velocities[0])
        print()


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

if __name__ == "__main__":
    main()

