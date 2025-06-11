import numpy as np
import scipy.fftpack as fft


def poisson_solver_periodic(rho, box_size, G=1.0):
    """
    Solve Poisson equation with periodic boundary conditions using FFT.

    Parameters
    ----------
    rho : ndarray
        3D array of mass density
    box_size : float
        Physical size of the simulation box
    G : float
        Gravitational constant (default 1)

    Returns
    -------
    phi : ndarray
        3D array of potential
    """
    N = rho.shape[0]  # assume cube: NxNxN
    kfreq = fft.fftfreq(N, d=box_size/N) * 2*np.pi
    kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2

    rho_k = fft.fftn(rho)
    phi_k = np.zeros_like(rho_k, dtype=complex)

    # Avoid division by zero
    k2[0,0,0] = 1.0

    phi_k = -4*np.pi*G * rho_k / k2

    # Restore k=0 mode to zero (mean potential arbitrary)
    phi_k[0,0,0] = 0.0

    phi = np.real(fft.ifftn(phi_k))
    return phi


def poisson_solver_periodic_safe(rho, box_size, soft_len, G=1.0):
    """
    Solve ∇²φ = 4πG ρ  on a cubic grid with periodic BCs.
    Setting soft_len>0 multiplies the k-space Green’s function by
    exp(-k² soft_len² / 2) → Gaussian force softening.
    """
    N     = rho.shape[0]
    rho_k = np.fft.fftn(rho)

    k     = 2.0 * np.pi * np.fft.fftfreq(N, d=box_size / N)
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    k2    = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = np.inf

    soft_factor = np.exp(-soft_len**2 * k2 / 2.0) if soft_len > 0.0 else 1.0
    phi_k = -4.0 * np.pi * G * soft_factor * rho_k / k2
    phi_k[0, 0, 0] = 0.0                     # mean(φ)=0

    return np.fft.ifftn(phi_k).real


# periodic approximate to isolated
def poisson_solver_isolated(rho, box_size, soft_len, G=1.0):
    """
    Solve Poisson equation with isolated boundary conditions using FFT-based convolution.

    Parameters
    ----------
    rho : ndarray
        3D array of mass density
    box_size : float
        Physical size of the simulation box
    G : float
        Gravitational constant (default 1)

    Returns
    -------
    phi : ndarray
        3D array of potential (same size as input rho)
    """
    N = rho.shape[0]  # assume cubic grid
    L = box_size
    dx = L / N

    # Pad density to 2N³ to avoid wrap-around
    N2 = 2 * N
    rho_pad = np.zeros((N2, N2, N2))

    # Center rho in the padded grid
    start = N//2-1
    end = start + N 
    rho_pad[start:end, start:end, start:end] = rho

    rho_k = np.fft.fftn(rho_pad)

    k     = 2.0 * np.pi * np.fft.fftfreq(N2, d=box_size / N)
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    k2    = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = np.inf

    soft_factor = np.exp(-soft_len**2 * k2 / 2.0) if soft_len > 0.0 else 1.0
    phi_k = -4.0 * np.pi * G * soft_factor * rho_k / k2
    phi_k[0, 0, 0] = 0.0                     # mean(φ)=0

    phi_pad = np.fft.ifftn(phi_k).real

    # Extract potential from central region
    phi = phi_pad[start:end, start:end, start:end]
    return phi

# real green function (still have bug)
def poisson_solver_isolated_green(rho, box_size, soft_len, G=1.0):
    """
    Solve Poisson equation with isolated boundary conditions using FFT-based convolution.

    Parameters
    ----------
    rho : ndarray
        3D array of mass density
    box_size : float
        Physical size of the simulation box
    G : float
        Gravitational constant (default 1)

    Returns
    -------
    phi : ndarray
        3D array of potential (same size as input rho)
    """
    N = rho.shape[0]  # assume cubic grid
    L = box_size
    dx = L / N

    # Pad density to 2N³ to avoid wrap-around
    N2 = 2 * N
    rho_pad = np.zeros((N2, N2, N2))

    # Center rho in the padded grid
    start = N//2-1
    end = start + N 
    rho_pad[start:end, start:end, start:end] = rho

    # Construct softened Green's function in real space
    x = np.fft.fftfreq(N2, d=dx) * N2
    x = np.where(x > N, x - N2, x) * dx  
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    r2 = X**2 + Y**2 + Z**2
    green = 1.0 / np.sqrt(r2 + soft_len**2)
    green *= G / (4 * np.pi)  # Correct prefactor for Poisson in 3D
    green = np.fft.ifftshift(green)  
    

    # FFT-based convolution
    rho_k = fft.fftn(rho_pad)
    green_k = fft.fftn(green)
    phi_k = rho_k * green_k
    phi_pad = np.real(fft.ifftn(phi_k))

    # Extract potential from central region
    phi = phi_pad[start:end, start:end, start:end]
    return phi
