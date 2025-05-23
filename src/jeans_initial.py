import numpy as np

# NFW profile
def nfw_density_profile(r, rho0=1.0, rs=1.0):
    return rho0 / ((r / rs) * (1 + r / rs)**2)

def nfw_potential_profile(r, G=1.0, rho0=1.0, rs=1.0):
    x = r / rs
    M_r = 4 * np.pi * rho0 * rs**3 * (np.log(1 + x) - x / (1 + x))
    return -G * M_r / r

# Plummer profile
def plummer_density_profile(r, M=1.0, a=1.0):
    return (3 * M) / (4 * np.pi * a**3) * (1 + (r / a)**2)**(-2.5)

def plummer_potential_profile(r, G=1.0, M=1.0, a=1.0):
    return -G * M / np.sqrt(r**2 + a**2)





