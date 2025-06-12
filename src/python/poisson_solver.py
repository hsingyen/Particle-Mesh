import numpy as np

def poisson_solver_periodic(rho, box_size, G_k, G=1.0):
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

    phi_k = -4.0 * np.pi * G  * rho_k / k2
    phi_k[0, 0, 0] = 0.0                    

    return np.fft.ifftn(phi_k).real


def poisson_solver_isolated(rho,G_k, N, box_size, G=1.0):
    dx = box_size / N
    padded_rho = np.zeros((2*N, 2*N, 2*N))
    padded_rho[0:N, 0:N, 0:N] = rho

    # Load or compute Green function
    #G_k = green(N, box_size)
    
    # FFT of density (with volume element)
    rho_k = np.fft.rfftn(padded_rho*(dx**3))

    # Convolution in Fourier space
    phi_k = rho_k * G_k

    # Inverse FFT (don't forget normalization)
    phi_padded = np.fft.irfftn(phi_k)
    #phi_padded *= (2*N)**1.5
    phi_padded /= dx

    # Multiply by 4πG
    #phi_padded *= 4 * np.pi * G

    # Take the physical part
    phi = phi_padded[0:N, 0:N, 0:N]
    return phi

def green(N, box_size):
    dx = box_size / N
    g = np.zeros((2*N, 2*N, 2*N))
    N2 = 2*N
    sqrt3N = np.sqrt(3) * N

    for i in range(N2):
        for j in range(N2):
            for k in range(N2):
                ii = min(i, 2*N - i)
                jj = min(j, 2*N - j)
                kk = min(k, 2*N - k)
                r = np.sqrt(ii**2 + jj**2 + kk**2)

                if r == 0:
                    g[i, j, k] = 0
                else:
                    g[i, j, k] = -1 / r
    #print(g)
    G_k = np.fft.rfftn(g)
    return G_k
