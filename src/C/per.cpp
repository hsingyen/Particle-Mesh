#include "orbit_integrator.hpp"
#include "jeans_initial.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <tuple>
#include <cmath>
#include <omp.h>

int flatten_index(const IndexTriple& idx, int N) {
    return idx.x * N * N + idx.y * N + idx.z;
}

int main() {
    int num_threads = 8; 
    //omp_set_num_threads(num_threads);
    std::cout << "Using " << num_threads << " threads." << std::endl;

    // === Simulation Parameters ===
    const int    N           = 256;
    const double box_size    = 1.0;
    const int    N_particles = 20000;
    const double dt          = 2e-4;
    const int    n_steps     = 200;
    const std::string dp     = "tsc";
    const std::string solver = "periodic";
    const std::string integrator = "dkd";
    const std::string mode   = "stable";
    const double a           = 0.005;
    const double G           = 1.0;

    // === Initialization ===
    ParticleArray positions, velocities;
    MassArray    masses;
    std::tie(positions, velocities, masses)
        = create_particles_single(N_particles, box_size, a, 1.0, mode, solver, G);

    // Precompute Green's function for isolated solver
    std::vector<std::complex<double>> G_k;
    if (solver == "isolated") {
        G_k = compute_green_ft(N, box_size);
    }

    // === Timing starts ===
    double start = omp_get_wtime();

    // === Time‐integration loop ===
    for (int step = 0; step < n_steps; ++step) {
        StepResult step_result;
        if      (integrator == "kdk") step_result = kdk_step (positions, velocities, masses, dt, N, box_size, dp, solver, G_k);
        else if (integrator == "dkd") step_result = dkd_step (positions, velocities, masses, dt, N, box_size, dp, solver, G_k);
        else /*rk4*/                  step_result = rk4_step(positions, velocities, masses, dt, N, box_size, dp, solver, G_k, G);

        positions = std::move(step_result.positions);
        velocities = std::move(step_result.velocities);
        masses    = std::move(step_result.masses);
    }

    double end = omp_get_wtime();

    std::cout << dp << "+" << integrator << "+" << solver << ": N = " << N << ", N_particles = " << N_particles << std::endl;
    std::cout << "Time taken: " << (end - start) << " seconds" << std::endl;
    std::cout << "Average per step: " << (end - start) / n_steps << " seconds" << std::endl;

    return 0;
}
