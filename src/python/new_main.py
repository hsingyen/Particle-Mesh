import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from new_orbit_integrator import kdk_step, dkd_step,rk4_step
from new_orbit_integrator import compute_phi
from mpl_toolkits.mplot3d import Axes3D
from jeans_initial import create_particles_single, create_particles_double
from poisson_solver import green

# === Simulation parameters ===
N = 128                    # Grid size: N x N x N
box_size = 1.0
N_particles =  100         # 10000
center = N // 2
dt = 2e-4
n_steps = 100              # 200
dp = 'cic'                 # 'ngp', 'cic', or 'tsc'
solver = 'periodic'        # 'isolated', 'periodic 
integrator = 'dkd'         # 'kdk' or 'dkd' or 'rk4' 
dx = box_size/N
mode = 'stable'
a = 0.01

if solver == 'isolated':
    G_k = green(N, box_size)
else:
    G_k =0

# === useful function ===
def potential_slice(phi, box_size, title):
    """Plot mid-plane slice of potential."""
    plt.imshow(phi[:,:,center], extent=[0, box_size, 0, box_size], origin='lower')
    plt.colorbar(label="Potential Φ")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def particle_slice_XY(positions, title):
    fig_init, ax_init = plt.subplots()
    ax_init.scatter(positions[:, 0], positions[:, 1], s=5, alpha=0.6)
    ax_init.set_aspect('equal')
    ax_init.set_xlim(0, box_size)
    ax_init.set_ylim(0, box_size)
    ax_init.set_xlabel("x")
    ax_init.set_ylabel("y")
    ax_init.set_title(f'Particle SLice_{title}(XYplane)')
    plt.tight_layout()
    plt.show()

def compute_total_momentum(velocities, masses):
    """Compute momentum (vector)"""
    return np.sum(masses[:, None] * velocities, axis=0)

def compute_total_energy(positions, velocities, masses, N, box_size,dp,solver, G_k):
    """Compute total energy (kinetic + potential) of the system."""
    KE = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))

    phi,weights = compute_phi(positions, masses, N, box_size, dp, solver, G_k)
    particle_values = []
    for weight in weights:
        phi_par = 0.0  # use scalar
        for idx, w in weight:
            phi_par += phi[idx] * w
        particle_values.append(phi_par)
    particle_values = np.array(particle_values)
    PE = 0.5*np.sum(masses*particle_values)
    return KE, PE, KE+PE

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


# === main simulation ===
def main():
    np.random.seed(42)
    # single plummer(mode = 'stable, contract, expand)
    positions, velocities, masses = create_particles_single(N_particles,box_size,a = a, M = 1.0, mode = mode ,solver = solver,G=1.0)
    # two plummer(mode, addv= True,False, v_offset=)
    #positions, velocities, masses = create_particles_double(N_particles,box_size,a = a, M = 1.0, mode=mode,solver = solver,G=1.0 , add_initial_velocity=True, v_offset= 0.1)
    particle_slice_XY(positions, 'Initial setup')
    # initial total energy/momentum
    # K_direct, P_direct, energy_direct = compute_energies_direct(positions, velocities, masses)
    K_poi, P_poi, energy_poisson = compute_total_energy(positions,velocities,masses, N, box_size, dp , solver,G_k)
    momentum = compute_total_momentum(velocities,masses)
    # Q_J_dir = 2 * K_direct / abs(P_direct)
    Q_J_poi = 2*K_poi/(abs(P_poi))
    # print(f"Initial Energy(direct) = {energy_direct:.4f}, Qj = {Q_J_dir:.4f}")
    print(f"Initial Energy(poisson) = {energy_poisson:.4f},Qj = {Q_J_poi:.4f}")
    print("Initial momentum =", momentum)

    # initial setup figure
    particle_slice_XY(positions, 'Initial setup')

    # lists for update date
    momentum_errors = []         # total error ||ΔP||
    momentum_errors_xyz = []     # per-axis error [|ΔPx|, |ΔPy|, |ΔPz|]

    frames = []           # potential field frames
    particle_frames = []  # particle position frames
    energies = []
    energies.append([K_poi,P_poi])
    saved_frames = set()

    # update
    for step in range(n_steps):
        if integrator == 'kdk':
            positions, velocities, masses, phi = kdk_step(positions, velocities, masses, dt, N, box_size, dp, solver,G_k)
        elif integrator == 'dkd':
            positions, velocities,masses, phi = dkd_step(positions, velocities, masses, dt, N, box_size, dp, solver,G_k)
        elif integrator == 'rk4':
            positions, velocities,masses, phi = rk4_step(positions, velocities, masses, dt, N, box_size, dp, solver,G_k)

        # compute energy/momentum each step
        KE, PE, total_energy = compute_total_energy(positions, velocities, masses, N, box_size,dp,solver,G_k)
        energies.append([KE, PE])
        current_momentum = compute_total_momentum(velocities, masses)
        delta_P = current_momentum - momentum
        momentum_errors.append(np.linalg.norm(delta_P))
        momentum_errors_xyz.append(np.abs(delta_P))

        # Save frames every 5 steps
        if step % 1 == 0:
            frames.append(phi[:,:,center].T.copy())
            particle_frames.append(positions.copy())
    
    PP_anim(frames, particle_frames, box_size, dp, solver)
    energy(energies)
    momentum_plot(momentum_errors_xyz)
        
# === figures ===
def PP_anim(frames, particle_frames, box_size,dp,solver):
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
    # save gif
    ani_combined.save(f"PP_anim_{dp}_{solver}.gif", writer='pillow')
    plt.show()

def energy(energies):
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

def momentum_plot(momentum_errors_xyz):
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
