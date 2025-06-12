import numpy as np
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

def create_particles_single(N_particles, box_size, a, M, mode, solver,G=1.0):
    boundary = solver
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

    # --- Step 3: Assign equal mass to all particles
    masses = np.full(N_particles, M / N_particles)

    if boundary == 'isoalated':
        mask = np.all((positions >= 0) & (positions < box_size), axis=1)
        positions = positions[mask]
        velocities = velocities[mask]
        masses = masses[mask]

    return positions, velocities, masses

def create_particles_double(N_particles, box_size, a, M, mode, solver,G, add_initial_velocity , v_offset):
    N_each = N_particles // 2
    positions_all = []
    velocities_all = []
    boundary = solver

    for center_shift, bulk_v in [(-0.1, +v_offset), (+0.1, -v_offset)]:
        box_center = np.array([box_size / 2 + center_shift, box_size/2, box_size/2])

        u = np.random.rand(N_each)
        r = a * (u**(-2/3) - 1)**(-0.5)

        theta = np.arccos(1 - 2 * np.random.rand(N_each))
        phi = 2 * np.pi * np.random.rand(N_each)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        positions = np.stack([x, y, z], axis=1) + box_center

        # velocity dispersion
        if mode == "stable":
            scale = 1.0
        elif mode == "contract":
            scale = 0.5
        elif mode == "expand":
            scale = 1.5
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        sigma = plummer_velocity_dispersion(r, a, M, G)
        velocities = np.random.normal(0, 1, size=(N_each, 3)) * (scale * sigma[:, np.newaxis])

        
        if add_initial_velocity:
            velocities[:, 0] += bulk_v 

        positions_all.append(positions)
        velocities_all.append(velocities)

    positions_all = np.vstack(positions_all)
    velocities_all = np.vstack(velocities_all)
    masses = np.full(N_particles, M / N_particles)

    if boundary == 'isoalated':
        mask = np.all((positions >= 0) & (positions < box_size), axis=1)
        positions = positions[mask]
        velocities = velocities[mask]
        masses = masses[mask]

    return positions_all, velocities_all, masses




