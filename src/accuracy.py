import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from new_orbit_integrator import kdk_step, dkd_step,rk4_step
from new_orbit_integrator import compute_phi
from utils import Timer  # optional
from mpl_toolkits.mplot3d import Axes3D
from jeans_initial import create_particles
import time

# === Simulation parameters ===
N = 128  # Grid size: N x N x N
box_size = 1.0
N_particles =  1000 #10000
center = N // 2
#dt = 0.001
#n_steps = 200  #200
#dp = 'ngp'  # 'ngp', 'cic', or 'tsc'
#solver = 'periodic' # 'isolated', 'periodic ,'periodic_safe'(softening = 0 equal to periodic)
#integrator = 'kdk'         # 'kdk' or 'dkd' or 'rk4' or 'hermite_individual'   or 'hermite_fixed'
self_force = False         # True or False
softening = 0.0 
a = 0.005

# === initial setup ====
def initial():
    np.random.seed(42)
    positions, velocities, masses = create_particles(N_particles, box_size, a = a , M =1.0, mode='stable',r_max = 5, G = 1.0)
    return positions, velocities, masses

# === evolution ===
def error_calculate(total_energy, initial_energy):
    return total_energy-initial_energy 

def update(n_steps, integrator, solver, dp, dt):
    for step in range(n_steps):
        # Orbit integration
        #print(step)
        ## change input parameter
        if integrator == 'kdk':
            positions, velocities, masses, phi = kdk_step(positions, velocities, masses, dt, N, box_size, dp, solver, subtract_self=self_force,soft_len=softening)
        elif integrator == 'dkd':
            positions, velocities,masses, phi = dkd_step(positions, velocities, masses, dt, N, box_size, dp, solver, subtract_self=self_force,soft_len=softening)
        elif integrator == 'rk4':
            positions, velocities,masses, phi = rk4_step(positions, velocities, masses, dt, N, box_size, dp, solver, subtract_self=self_force,soft_len=softening)
        
        error = error_calculate()
    return error
    

# === mass deposition test ===
def accuracy_for_massdp():
    dps = ['ngp', 'cic' ,'tsc']
    integrator = 'kdk'      #fixed integrator
    solver = 'periodic'
    n_steps = 200
    dts = [0.01,0.005,0.001,0.0005]
    errors = []
    fig,ax = plt.figure()
    plt.xlabel('dt')
    plt.ylabel('error')
    plt.xscale('log')
    plt.yscale('log')
    for dp in dps:
        for dt in dts:
            positions, velocities, masses = initial()
            error = update(n_steps, integrator, solver, dp , dt)
            errors.append(error)
        ax.plot(dts, errors, '-', label = f'{dp}')
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_massdp.png")
    plt.show()

# ===  integrator test ===
def accuracy_for_integrator():
    dp = 'ngp'
    integrators = ['kdk', 'dkd', 'rk4']
    solver = 'periodic'
    n_steps = 200
    dts = [0.01,0.005,0.001,0.0005]
    errors = []
    fig,ax = plt.figure()
    plt.xlabel('dt')
    plt.ylabel('error')
    plt.xscale('log')
    plt.yscale('log')
    for integrator in integrators:
        for dt in dts:
            positions, velocities, masses = initial()
            error = update(n_steps, integrator, solver, dp , dt)
            errors.append(error)
        ax.plot(dts, errors, '-', label =f'{integrator}')
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_integr.png")
    plt.show()

# === performance ===
def perfomance():
    dps = ['ngp', 'cic' ,'tsc']
    integrators = ['kdk', 'dkd', 'rk4']
    solver = 'periodic'
    dt = 0.001
    n_steps = 200
    N_particless = [1000, 5000, 10000, 50000, 100000]
    times = []
    fig,ax = plt.figure()
    plt.xlabel('number of particles')
    plt.ylabel('wall-clock time')
    for dp in dps:
        for integrator in integrators:
            for N_particles in N_particless:
                positions, velocities, masses = initial(N_particles)
                start_time = time.time()
                error = update(N_particless,integrator, solver, dp, dt)
                end_time = time.time()
                times.append(end_time-start_time)
            ax.plot(N_particless, times, '-', label =f'{dp}+{integrator}')
    plt.legend()
    plt.tight_layout()
    plt.savefig("performance.png")
    plt.show()


    #accuracy_for_massdp()
    #accuracy_for_integrator()
    #perfomance()


    
