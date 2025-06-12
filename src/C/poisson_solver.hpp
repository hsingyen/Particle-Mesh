#pragma once

#include <vector>

/**
 * Solve the Poisson equation on a 3D grid using FFT-based methods.
 * All functions expect a flattened density array of size N^3 (row-major: i*N*N + j*N + k).
 */

/**
 * Periodic Poisson solver: ∇²φ = 4πGρ with periodic boundary conditions via FFT.
 * @param rho_flat Flattened N^3 density array
 * @param N        Grid size per dimension
 * @param box_size Physical size of the cubic box
 * @param G        Gravitational constant (default 1.0)
 * @return         Flattened N^3 potential array φ
 */
std::vector<double> poisson_solver_periodic(
    const std::vector<double>& rho_flat,
    int N,
    double box_size,
    double G = 1.0);

/**
 * Periodic Poisson solver with Gaussian softening in k-space.
 * @param rho_flat Flattened N^3 density array
 * @param N        Grid size per dimension
 * @param box_size Physical size of the cubic box
 * @param soft_len Softening length for Gaussian filter
 * @param G        Gravitational constant (default 1.0)
 * @return         Flattened N^3 potential array φ
 */
std::vector<double> poisson_solver_periodic_safe(
    const std::vector<double>& rho_flat,
    int N,
    double box_size,
    double soft_len,
    double G = 1.0);

/**
 * Isolated Poisson solver via zero-padded FFT convolution.
 * @param rho_flat Flattened N^3 density array
 * @param N        Grid size per dimension
 * @param box_size Physical size of the cubic box
 * @param soft_len Softening length for Gaussian filter
 * @param G        Gravitational constant (default 1.0)
 * @return         Flattened N^3 potential array φ
 */
std::vector<double> poisson_solver_isolated(
    const std::vector<double>& rho_flat,
    int N,
    double box_size,
    double soft_len,
    double G = 1.0);
