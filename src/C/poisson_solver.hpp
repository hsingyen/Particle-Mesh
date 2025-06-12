// poisson_solver.hpp
#ifndef POISSON_SOLVER_HPP
#define POISSON_SOLVER_HPP

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <vector>
#include <complex>

// Solve ∇²φ = 4πG ρ with periodic BCs on an N×N×N grid
// rho: input density, size = N^3
// box_size: physical box length
// G: gravitational constant
// returns φ, size = N^3
std::vector<double> poisson_solver_periodic(
    const std::vector<double>& rho,
    int N,
    double box_size,
    double G = 1.0
);

// Solve ∇²φ = 4πG ρ for an isolated mass distribution via zero-padding
// rho: input density, size = N^3
// G_k: precomputed Green’s function FT, size = (2N)^2×(2N)×(N+1) complex-packed
// box_size: physical box length
// returns φ on original N^3 grid
std::vector<double> poisson_solver_isolated(
    const std::vector<double>& rho,
    const std::vector<std::complex<double>>& G_k,
    int N,
    double box_size,
    double G = 1.0
);

// Compute the Green’s function FT for isolated BCs on a 2N grid
// returns G_k of size (2N)^2×(2N)×(N+1) complex-packed
std::vector<std::complex<double>> compute_green_ft(
    int N,
    double box_size
);

#endif // POISSON_SOLVER_HPP
