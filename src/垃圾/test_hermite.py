import numpy as np
from orbit_integrator import hermite_individual_step

G = 1.0
T_max = 6.28   # One orbit (2π)
positions = []

# Two-particle setup
particles = [
    {
        'r': np.array([0.0, 0.0, 0.0]),
        'v': np.array([0.0, 0.0, 0.0]),
        'a': np.zeros(3),
        'j': np.zeros(3),
        't': 0.0,
        'dt': 0.01,
        'm': 1.0
    },
    {
        'r': np.array([1.0, 0.0, 0.0]),
        'v': np.array([0.0, 1.0, 0.0]),
        'a': np.zeros(3),
        'j': np.zeros(3),
        't': 0.0,
        'dt': 0.01,
        'm': 0.001
    }
]

# Compute initial acceleration and jerk
def compute_initial_forces(particles, G):
    for i in range(len(particles)):
        a = np.zeros(3)
        j = np.zeros(3)
        r_i = particles[i]['r']
        v_i = particles[i]['v']
        for jdx in range(len(particles)):
            if i == jdx:
                continue
            r_j = particles[jdx]['r']
            v_j = particles[jdx]['v']
            m_j = particles[jdx]['m']
            dr = r_j - r_i
            dv = v_j - v_i
            r2 = np.dot(dr, dr) + 1e-10
            r3 = r2 * np.sqrt(r2)
            r5 = r2 * r3
            a += G * m_j * dr / r3
            j += G * m_j * (dv / r3 - 3 * np.dot(dr, dv) * dr / r5)
        particles[i]['a'] = a
        particles[i]['j'] = j

compute_initial_forces(particles, G)

# Main loop: step until the orbiting particle reaches T_max
while particles[1]['t'] < T_max:
    hermite_individual_step(particles, G)
    positions.append(particles[1]['r'].copy())

# Print positions
for i, pos in enumerate(positions):
    print(f"Step {i}: r = {pos}")