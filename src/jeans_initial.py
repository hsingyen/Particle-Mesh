import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.stats import norm

# Plummer profile
def plummer_density_profile(r, a, M=1.0):
    return (3 * M) / (4 * np.pi * a**3) * (1 + (r / a)**2)**(-2.5)

def dphi_dr_plummer(r, a,G=1.0, M=1.0):
    return G * M * r / np.sqrt(r**2 + a**2)**3.0

def phi_plummer(r, a,G=1.0, M=1.0):
    return -G*M/np.sqrt(r**2+a**2)

def plummer_velocity_dispersion(r,a,M=1.0,G=1.0):
    return np.sqrt(G * M / np.sqrt(r**2 + a**2)/6.0)

def sample_plummer_radius(N, a, r_max=50.0):
    r = []
    while len(r) < N:
        u = np.random.rand()
        val = a * u**(2/3) / (1 - u**(2/3))**0.5
        if val < r_max * a:
            r.append(val)
    return np.array(r)

def create_particles(N_particles, box_size, a, M, mode="stable",r_max=5.0,G=1.0):
    """

    Parameters:
        N_particles       : int    
        box_size: float  
        a       : float 
        M       : float  
        G       : float 
        mode    : str    "stable", "contract", or "expand"

    Returns:
        positions: (N, 3) 
        velocities: (N, 3) 
        masses: (N,) 
    """
    # --- Step 1: Generate positions using Plummer distribution
    box_center = np.array([box_size / 2] * 3)
    u = np.random.rand(N_particles)
    r = a * (u**(-2/3) - 1)**(-0.5)  # radial distribution from inverse CDF

    theta = np.arccos(1 - 2 * np.random.rand(N_particles))
    phi = 2 * np.pi * np.random.rand(N_particles)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    positions = np.stack([x, y, z], axis=1) + box_center

    # --- Step 2: Compute velocity dispersion at each radius
    if mode == "stable":
        scale = 1.0
    elif mode == "contract":
        scale = 0.5
    elif mode == "expand":
        scale = 1.5
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    sigma = plummer_velocity_dispersion(r, a, M, G)
    velocities = np.random.normal(0, 1, size=(N_particles, 3)) * (scale * sigma[:, np.newaxis])
    # velocities = np.random.normal(
    #     loc=0.0,
    #     scale=scale*sigma[:,None],   # broadcast to (N,3)
    #     size=(N_particles,3)
    # )
    print("Hello")

    # --- Step 3: Assign equal mass to all particles
    masses = np.full(N_particles, M / N_particles)

    return positions, velocities, masses

def compute_enclosed_density(box_size,positions, masses, a, nbins=100):
    center = np.array([box_size / 2] * 3)
    rel_pos = positions - center
    r = np.linalg.norm(rel_pos, axis=1)

    r_bins = np.linspace(1e-4, np.max(r), nbins)
    rho_enclosed = []
    r_over_a = []

    for r_cut in r_bins:
        enclosed = r <= r_cut
        M_enclosed = np.sum(masses[enclosed])
        V = (4/3) * np.pi * r_cut**3
        rho_enclosed.append(M_enclosed / V)
        r_over_a.append(r_cut / a)

    return np.array(r_over_a), np.array(rho_enclosed)

def compute_shell_density(box_size, positions, masses, a, nbins=100):
    center = np.array([box_size / 2] * 3)
    rel_pos = positions - center
    r = np.linalg.norm(rel_pos, axis=1)

    r_edges = np.linspace(0, np.max(r), nbins + 1)
    rho_shell = []
    r_mid = []

    for i in range(nbins):
        r_inner = r_edges[i]
        r_outer = r_edges[i+1]
        in_shell = (r >= r_inner) & (r < r_outer)
        M_shell = np.sum(masses[in_shell])
        V_shell = (4/3) * np.pi * (r_outer**3 - r_inner**3)
        rho_shell.append(M_shell / V_shell if V_shell > 0 else 0)
        r_mid.append(0.5 * (r_inner + r_outer))

    return np.array(r_mid) / a, np.array(rho_shell)

def plot_density_vs_r(box_size, positions, masses, a, M):
    #r_over_a, rho_enclosed = compute_enclosed_density(box_size, positions, masses, a)
    r_over_a, rho_enclosed = compute_shell_density(box_size, positions, masses, a)
    r_plot = np.linspace(0.001, 2, 200)
    rho_true = (3 * M) / (4 * np.pi * a**3) * (1 + (r_plot / a)**2)**(-2.5)

    plt.figure(figsize=(6, 5))
    plt.plot(r_over_a, rho_enclosed, 'o', label="Simulated setup")
    plt.plot(r_plot / a, rho_true, '-', label="Plummer Analytic")
    plt.xscale('log')
    plt.yscale('log')
    #plt.xlim(0.001, 2)
    #plt.ylim(0,10)
    plt.xlabel(r'$r/a$')
    plt.ylabel(r'$\rho(r)$')
    plt.legend()
    plt.title("Plummer Density Profile")
    plt.tight_layout()
    plt.show()

def plot_velocity_distribution(velocities):
    plt.figure(figsize=(12, 4))
    for i, comp in enumerate(['vx', 'vy', 'vz']):
        plt.subplot(1, 3, i+1)
        data = velocities[:, i]
        mu, std = np.mean(data), np.std(data)
        x = np.linspace(data.min(), data.max(), 100)
        plt.hist(data, bins=40, density=True, alpha=0.6, label='Data')
        plt.plot(x, norm.pdf(x, mu, std), 'r--', label=f'Gaussian\n$\mu$={mu:.2f}, $\sigma$={std:.2f}')
        plt.xlabel(comp)
        plt.ylabel("PDF")
        plt.legend()
    plt.suptitle("Velocity Component Distributions")
    plt.tight_layout()
    plt.show()

def plot_velocity_dispersion_profile(positions, velocities, a, M=1.0, G=1.0, nbins=30):
    # Compute radius from center
    center = np.mean(positions, axis=0)
    rel_pos = positions - center
    r = np.linalg.norm(rel_pos, axis=1)
    
    # Use only vx for dispersion (could also average over vx, vy, vz)
    vx = velocities[:, 0]
    
    # Bin data in radius
    bins = np.logspace(np.log10(np.min(r[r>0])), np.log10(np.max(r)), nbins+1)
    r_bin_centers = 0.5 * (bins[:-1] + bins[1:])
    sigma_sim = []

    for i in range(nbins):
        mask = (r >= bins[i]) & (r < bins[i+1])
        if np.sum(mask) > 1:
            sigma = np.std(vx[mask])
            sigma_sim.append(sigma)
        else:
            sigma_sim.append(np.nan)
    
    sigma_sim = np.array(sigma_sim)

    # Compute analytic dispersion
    r_analytic = np.logspace(np.log10(np.min(r_bin_centers[~np.isnan(sigma_sim)])), np.log10(np.max(r_bin_centers)), 200)
    sigma_analytic = plummer_velocity_dispersion(r_analytic, a, M, G)

    # Plotting
    plt.figure(figsize=(7,5))
    plt.plot(r_analytic/a, sigma_analytic, label="Plummer Analytic", color='orange')
    plt.scatter(r_bin_centers/a, sigma_sim, label="Simulated dispersion(r)", s=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$r/a$')
    plt.ylabel(r'$\sigma(r)$')
    plt.title("Velocity Dispersion Profile")
    plt.legend()
    plt.grid(True, which="both", ls='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
