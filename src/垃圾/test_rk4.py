# test_rk4.py  – minimal two-body sanity check for the PM RK4 integrator
import numpy as np
import matplotlib.pyplot as plt
from poisson_solver import poisson_solver_periodic_safe
from orbit_integrator import rk4_step
from mass_deposition  import deposit_cic        # swap for deposit_ngp / deposit_tsc if desired

def direct_energy(pos, vel, mass, G=1.0):
    r_vec = pos[1] - pos[0]
    r     = np.linalg.norm(r_vec)
    K     = 0.5 * mass[1] * np.dot(vel[1], vel[1])
    U     = -G * mass[0] * mass[1] / r
    return K + U

# ---------------------------------------------------------------------
def two_body_test():
    # ---- mesh / time-step parameters --------------------------------
    G        = 1.0
    box_size = 10.0        # push periodic images far away
    N        = 64         # grid (dx ≈ 0.04)
    dt       = 0.02       # ensure v_max dt < ½ dx
    n_steps  = 500

    # ---- particle properties ----------------------------------------
    mass = np.array([1.0, 0.001])                # primary, satellite

    r0            = 1.0
    primary_pos   = np.array([box_size/2]*3)
    satellite_pos = primary_pos + np.array([r0, 0.0, 0.0])

    v_circ        = np.sqrt(G * mass[0] / r0)    # circular velocity
    primary_vel   = np.zeros(3)
    satellite_vel = np.array([0.0, v_circ, 0.0])


    # stack into (2,3) arrays
    pos = np.vstack((primary_pos, satellite_pos))
    vel = np.vstack((primary_vel,  satellite_vel))

    # ---- storage for trajectory -------------------------------------
    traj = np.empty((n_steps, 3))

    # ---- main integration loop --------------------------------------
    for i in range(n_steps):
        traj[i] = pos[1]                        # record satellite
        pos, vel = rk4_step(pos, vel, mass,
                            dt, N, box_size,
                            deposit_cic,
                            soft_len = 0.05,    # 0 → raw force
                            G = G)
        print(direct_energy(pos, vel, mass, G=G))  # sanity check


    # ---- plot --------------------------------------------------------
    plt.figure(figsize=(5,5))
    plt.plot(traj[:,0], traj[:,1], lw=1.2)
    plt.scatter(primary_pos[0], primary_pos[1],
                c='k', s=50, marker='*', zorder=3, label='primary')
    plt.gca().set_aspect('equal')
    plt.xlim(primary_pos[0]-1.5, primary_pos[0]+1.5)
    plt.ylim(primary_pos[1]-1.5, primary_pos[1]+1.5)
    plt.xlabel('x');  plt.ylabel('y')
    plt.title('2-body orbit on a periodic PM mesh')
    plt.grid(True);   plt.legend()
    plt.show()

# ---------------------------------------------------------------------
if __name__ == '__main__':
    two_body_test()