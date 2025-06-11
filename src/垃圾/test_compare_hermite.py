import numpy as np
import matplotlib.pyplot as plt
import copy

from orbit_integrator import hermite_step_fixed, hermite_individual_step

G = 1.0
dt = 0.01
T_max = 6.3  # a bit over one full orbit (2π)

# Initial two-body system
initial_particles = [
    {
        'r': np.array([0.0, 0.0, 0.0]),
        'v': np.array([0.0, 0.0, 0.0]),
        'a': np.zeros(3),
        'j': np.zeros(3),
        't': 0.0,
        'dt': dt,
        'm': 1.0
    },
    {
        'r': np.array([1.0, 0.0, 0.0]),
        'v': np.array([0.0, 1.0, 0.0]),
        'a': np.zeros(3),
        'j': np.zeros(3),
        't': 0.0,
        'dt': dt,
        'm': 0.001
    }
]

# Compute initial forces and jerks
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

# Clone particles
particles_fixed = copy.deepcopy(initial_particles)
particles_indiv = copy.deepcopy(initial_particles)

# Compute initial forces
compute_initial_forces(particles_fixed, G)
compute_initial_forces(particles_indiv, G)

# Simulate with fixed step
pos_fixed = []
for step in range(int(T_max / dt)):
    hermite_step_fixed(particles_fixed, G)
    pos_fixed.append(particles_fixed[1]['r'].copy())

# Simulate with individual time step
pos_indiv = []
while particles_indiv[1]['t'] < T_max:
    hermite_individual_step(particles_indiv, G)
    pos_indiv.append(particles_indiv[1]['r'].copy())

# Plot
pos_fixed = np.array(pos_fixed)
pos_indiv = np.array(pos_indiv)

plt.plot(pos_fixed[:,0], pos_fixed[:,1], label='Fixed Hermite', lw=1)
plt.plot(pos_indiv[:,0], pos_indiv[:,1], label='Individual Hermite', lw=1, linestyle='--')
plt.gca().set_aspect('equal')
plt.title("Comparison: Fixed vs Individual Time Step Hermite")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

# Print final positions
print("Final position (fixed):     ", particles_fixed[1]['r'])
print("Final position (individual):", particles_indiv[1]['r'])
print("Position error:", np.linalg.norm(particles_fixed[1]['r'] - particles_indiv[1]['r']))