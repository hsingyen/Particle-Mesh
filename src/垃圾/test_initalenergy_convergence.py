import numpy as np
import matplotlib.pyplot as plt
from jeans_initial import create_particles
from new_orbit_integrator import compute_phi

softening = 0.005
box_size = 1.0
N_particles = 1000
dp = 'cic'         # 'ngp', 'cic', or 'tsc'
solver = 'periodic_safe'

def compute_PE_direct(x, m, G=1.0, softening=1e-5):
    N = len(m)
    PE = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            r2 = np.dot(dx, dx) + softening**2
            PE -= G * m[i] * m[j] / np.sqrt(r2)
    return PE

def compute_PE_pm(x, m, N_grid, box_size, dp, solver):
    phi, weights = compute_phi(x, m, N_grid, box_size, dp, solver, soft_len=softening)
    particle_phi = []
    for wlist in weights:
        val = 0.0
        for idx, weight in wlist:
            val += phi[idx] * weight
        particle_phi.append(val)
    particle_phi = np.array(particle_phi)
    PE = 0.5 * np.sum(m * particle_phi)
    return PE



def main():
    print("Generating particles...")
    x, v, m = create_particles(
        N_particles, box_size,
        profile='plummer',
        velocity_mode='stable',
        velocity_distribution='isotropic'
    )

    print("Computing direct PE...")
    PE_direct = compute_PE_direct(x, m, softening=softening)
    print(f"Direct PE = {PE_direct:.6e}")

    grid_sizes = [16, 32, 64, 128, 256]
    relative_errors = []

    for N in grid_sizes:
        PE_pm = compute_PE_pm(x, m, N, box_size, dp, solver)
        rel_err = abs(PE_pm - PE_direct) / abs(PE_direct)
        relative_errors.append(rel_err)
        print(f"N = {N:4d} → PE_pm = {PE_pm:.6e}, rel error = {rel_err:.2e}")

    plt.figure()
    plt.plot(grid_sizes, relative_errors, 'o-', lw=2)
    plt.xlabel("Grid Size (N)")
    plt.ylabel("Relative PE Error")
    plt.yscale("log")
    plt.grid(True)
    plt.title(f"Convergence of PM Potential Energy ({dp.upper()} deposition)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()