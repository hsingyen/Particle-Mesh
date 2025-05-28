import numpy as np
from scipy.integrate import quad

# NFW profile
def nfw_density_profile(r, rho0=1.0, rs=1.0):
    return rho0 / ((r / rs) * (1 + r / rs)**2)

def dphi_dr_nfw(r, G=1.0, rho0=1.0, rs=1.0):
    x = r / rs
    M_r = 4 * np.pi * rho0 * rs**3 * (np.log(1 + x) - x / (1 + x))
    return G * M_r / r**2

# Plummer profile
def plummer_density_profile(r, M=1.0, a=1.0):
    return (3 * M) / (4 * np.pi * a**3) * (1 + (r / a)**2)**(-2.5)

def dphi_dr_plummer(r, G=1.0, M=1.0, a=1.0):
    return G * M * r / (r**2 + a**2)**(1.5)

# Jeans equation solver
def jeans_velocity_dispersion(r, density_func, dphi_dr_func, r_max=10.0, **kwargs):
    sigma2 = []
    for ri in r:
        integrand = lambda rp: density_func(rp, **kwargs) * dphi_dr_func(rp, **kwargs)
        integral, _ = quad(integrand, ri, r_max)
        sigma2_i = integral / density_func(ri, **kwargs)
        sigma2.append(sigma2_i)
    return np.array(sigma2)

# Position sampler
def sample_positions_from_density(N, r_max, density_func, **kwargs):
    positions = []
    r_vals = np.linspace(0.001, r_max, 1000)
    rho_vals = density_func(r_vals, **kwargs)
    rho_max = np.max(rho_vals)

    while len(positions) < N:
        r_try = np.random.uniform(0, r_max)
        prob = density_func(r_try, **kwargs) / rho_max
        if np.random.rand() < prob:
            theta = np.arccos(1 - 2 * np.random.rand())
            phi = 2 * np.pi * np.random.rand()
            x = r_try * np.sin(theta) * np.cos(phi)
            y = r_try * np.sin(theta) * np.sin(phi)
            z = r_try * np.cos(theta)
            positions.append([x, y, z])
    return np.array(positions)

# Common particle creator
def create_particles(N_particles, box_size, profile="plummer", velocity_mode="stable", velocity_distribution="isotropic"):
    G = 1.0
    r_max = box_size / 5
    box_center = np.array([box_size / 2] * 3)

    if profile == "plummer":
        M = 1.0
        a = 1.0
        density_func = plummer_density_profile
        dphi_func = dphi_dr_plummer
        kwargs = dict(M=M, a=a)
    elif profile == "nfw":
        rho0 = 1.0
        rs = 1.0
        density_func = nfw_density_profile
        dphi_func = dphi_dr_nfw
        kwargs = dict(rho0=rho0, rs=rs)
    else:
        raise ValueError(f"Unknown profile: {profile}")

    positions = sample_positions_from_density(N_particles, r_max, density_func, **kwargs)
    positions += box_center
    r_sample = np.linalg.norm(positions - box_center, axis=1)

    sigma2_vals = jeans_velocity_dispersion(r_sample, density_func, dphi_func, r_max=10.0, **kwargs)
    sigma_vals = np.sqrt(sigma2_vals)

    if velocity_mode == "stable":
        scale = 1.0
    elif velocity_mode == "contract":
        scale = 0.1
    elif velocity_mode == "expand":
        scale = 2.0
    else:
        raise ValueError(f"Unknown velocity_mode: {velocity_mode}")

    if velocity_distribution == "isotropic":
        velocities = np.random.normal(0, 1, size=(N_particles, 3)) * (scale * sigma_vals[:, np.newaxis])
    elif velocity_distribution == "radial":
        directions = positions - box_center
        unit_vectors = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
        v_mags = np.random.normal(0, 1, size=N_particles) * (scale * sigma_vals)
        velocities = unit_vectors * v_mags[:, np.newaxis]
    else:
        raise ValueError(f"Unknown velocity_distribution: {velocity_distribution}")

    masses = np.ones(N_particles) * (1.0 / N_particles)
    return positions, velocities, masses
