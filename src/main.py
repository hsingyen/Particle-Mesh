import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from poisson_solver import poisson_solver_periodic
from mass_deposition import deposit_ngp, deposit_cic, deposit_tsc
from orbit_integrator import kdk_step, dkd_step
from utils import Timer  # optional
from mpl_toolkits.mplot3d import Axes3D

# === Simulation parameters ===
N = 64  # Grid size: N x N x N
box_size = 1.0
N_particles = 1000
center = N // 2
dt = 0.01
n_steps = 100
deposition_scheme = 'cic'  # 'ngp', 'cic', or 'tsc'
integrator = 'kdk'         # 'kdk' or 'dkd'

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


def compute_particle_density(positions, N, box_size):
    """Compute 2D density field by counting particles per (x,y) grid cell."""
    density = np.zeros((N, N))
    dx = box_size / N

    for pos in positions:
        ix = int(pos[0] / dx) % N
        iy = int(pos[1] / dx) % N
        density[ix, iy] += 1

    return density

# === Main Simulation ===
def main():
    np.random.seed(42)

    # --- Static tests ---
    rho_point = create_point_mass(N)
    phi_point = poisson_solver_periodic(rho_point, box_size)
    plot_slice(phi_point, box_size, "Mid-plane Potential (Point Mass)")

    positions_static, velocities_static, masses_static = create_random_particles(N_particles, box_size)

    if deposition_scheme == 'ngp':
        rho_static = deposit_ngp(positions_static, masses_static, N, box_size)
    elif deposition_scheme == 'cic':
        rho_static = deposit_cic(positions_static, masses_static, N, box_size)
    elif deposition_scheme == 'tsc':
        rho_static = deposit_tsc(positions_static, masses_static, N, box_size)
    else:
        raise ValueError("Unknown deposition scheme")

    phi_static = poisson_solver_periodic(rho_static, box_size)
    plot_slice(phi_static, box_size, "Mid-plane Potential (Random Particles)")

    # --- Time-evolution simulation ---
    positions, velocities, masses = create_random_particles(N_particles, box_size)

    frames = []           # potential field frames
    particle_frames = []  # particle position frames
    energies = []
    density_frames = []

    for step in range(n_steps):
        # Mass deposition
        if deposition_scheme == 'ngp':
            rho = deposit_ngp(positions, masses, N, box_size)
        elif deposition_scheme == 'cic':
            rho = deposit_cic(positions, masses, N, box_size)
        elif deposition_scheme == 'tsc':
            rho = deposit_tsc(positions, masses, N, box_size)

        # Poisson solve
        phi = poisson_solver_periodic(rho, box_size)

        # Save energy
        KE, PE = compute_total_energy(positions, velocities, masses, phi, N, box_size)
        energies.append((KE, PE))

        # Save frames every 2 steps
        if step % 2 == 0:
            frames.append(phi[:,:,center].copy())
            particle_frames.append(positions.copy())
            density = compute_particle_density(positions, N, box_size)
            density_frames.append(density.copy())

        # Orbit integration
        if integrator == 'kdk':
            positions, velocities = kdk_step(positions, velocities, masses, dt, phi, N, box_size)
        elif integrator == 'dkd':
            positions, velocities = dkd_step(positions, velocities, masses, dt, phi, N, box_size)
        else:
            raise ValueError("Unknown integrator!")


    # --- Potential Field Animation ---
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], extent=[0, box_size, 0, box_size], origin='lower',
                   vmin=np.min(frames[0]), vmax=np.max(frames[0]))
    title = ax.set_title(f"Potential at Frame 0")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    plt.tight_layout()

    def animate_potential(i):
        im.set_data(frames[i])
        title.set_text(f"Potential at Frame {i}")
        return [im, title]

    ani = animation.FuncAnimation(fig, animate_potential, frames=len(frames), interval=100, blit=False)
    plt.show()

    # --- Particle Motion Animation ---
    fig2, ax2 = plt.subplots()
    scatter = ax2.scatter(particle_frames[0][:,0], particle_frames[0][:,1], s=1)
    ax2.set_xlim(0, box_size)
    ax2.set_ylim(0, box_size)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    title2 = ax2.set_title(f"Particles at Frame 0")
    plt.tight_layout()

    def animate_particles(i):
        scatter.set_offsets(particle_frames[i])
        title2.set_text(f"Particles at Frame {i}")
        return scatter, title2

    ani_particles = animation.FuncAnimation(fig2, animate_particles, frames=len(particle_frames), interval=100, blit=False)
    plt.show()

    # --- Combined Potential + Particles Animation ---
    fig3, ax3 = plt.subplots()
    im = ax3.imshow(frames[0], extent=[0, box_size, 0, box_size], origin='lower',
                    vmin=np.min(frames[0]), vmax=np.max(frames[0]), cmap='viridis')
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

    ani_combined = animation.FuncAnimation(fig3, animate_combined, frames=len(frames), interval=100, blit=False)
    plt.show()

    # --- Particle Density Field Animation ---
    fig4, ax4 = plt.subplots()
    im_density = ax4.imshow(density_frames[0].T, extent=[0, box_size, 0, box_size], origin='lower', cmap='inferno')
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    title4 = ax4.set_title(f"Particle Density at Frame 0")
    plt.tight_layout()

    def animate_density(i):
        im_density.set_data(density_frames[i].T)
        title4.set_text(f"Particle Density at Frame {i}")
        return im_density, title4

    ani_density = animation.FuncAnimation(fig4, animate_density, frames=len(density_frames), interval=100, blit=False)

    plt.show()

    # --- 3D Particle Scatter Animation ---
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111, projection='3d')
    scatter3d = ax5.scatter(particle_frames[0][:,0], particle_frames[0][:,1], particle_frames[0][:,2], s=1)

    ax5.set_xlim(0, box_size)
    ax5.set_ylim(0, box_size)
    ax5.set_zlim(0, box_size)
    ax5.set_xlabel("x")
    ax5.set_ylabel("y")
    ax5.set_zlabel("z")
    title5 = ax5.set_title(f"3D Particles at Frame 0")
    plt.tight_layout()

    def animate_particles_3d(i):
        scatter3d._offsets3d = (particle_frames[i][:,0], particle_frames[i][:,1], particle_frames[i][:,2])
        title5.set_text(f"3D Particles at Frame {i}")
        #ax5.view_init(elev=30., azim=i)
        return scatter3d, title5

    ani_particles_3d = animation.FuncAnimation(fig5, animate_particles_3d, frames=len(particle_frames), interval=100, blit=False)

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

if __name__ == "__main__":
    main()