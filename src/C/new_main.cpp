#include "orbit_integrator.hpp"
#include "jeans_initial.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <tuple>
#include <cmath>
#include <chrono>
#include <omp.h>

int flatten_index(const IndexTriple& idx, int N) {
    return idx.x * N * N + idx.y * N + idx.z;
}

int main() {
    // === Simulation Parameters ===
    const int    N           = 128;
    const double box_size    = 1.0;
    const int    N_particles = 5000;
    const double dt          = 2e-4;
    const int    n_steps     = 100;
    const std::string dp     = "ngp";
    const std::string solver = "isolated";
    const std::string integrator = "kdk";
    const std::string mode   = "stable";
    const double a           = 0.005;
    const double G           = 1.0;
    std::cout << "Using " << omp_get_max_threads() << " OpenMP threads\n";

    // === Initialization ===
    ParticleArray positions, velocities;
    MassArray    masses;
    std::tie(positions, velocities, masses)
        = create_particles_single(N_particles, box_size, a, 1.0, mode, solver, G);

    // Open CSV outputs
    std::ofstream log_file("simulation_output.csv");
    log_file << "step,KE,PE,Total,Px,Py,Pz\n";

    std::ofstream pos_file("particle_positions.csv");
    pos_file << "step,id,x,y,z\n";

    // Clear any old potential dump
    {
        std::ofstream clear_file("potential_N128.csv", std::ios::trunc);
    }

    // Precompute Green's function for isolated solver
    std::vector<std::complex<double>> G_k;
    if (solver == "isolated") {
        G_k = compute_green_ft(N, box_size);
    }

    //time measurement
    double total_integrator_time = 0.0;
    // === Time‐integration loop ===
    for (int step = 0; step < n_steps; ++step) {

        auto t0 = std::chrono::high_resolution_clock::now();
        // 1) Advance one step
        StepResult step_result;
        if      (integrator == "kdk") step_result = kdk_step (positions, velocities, masses, dt, N, box_size, dp, solver, G_k);
        else if (integrator == "dkd") step_result = dkd_step (positions, velocities, masses, dt, N, box_size, dp, solver, G_k);
        else /*rk4*/                  step_result = rk4_step(positions, velocities, masses, dt, N, box_size, dp, solver, G_k, G);
        auto t1 = std::chrono::high_resolution_clock::now();
        total_integrator_time += std::chrono::duration<double>(t1 - t0).count();


        // 2) Unpack the survivors
        positions = std::move(step_result.positions);
        velocities = std::move(step_result.velocities);
        masses    = std::move(step_result.masses);
        size_t M = positions.size();  // actual number of particles remaining

        // 3) Kinetic energy and total momentum
        double KE = 0.0;
        std::array<double,3> P = {0.0, 0.0, 0.0};
        for (size_t i = 0; i < M; ++i) {
            const auto& v = velocities[i];
            KE += 0.5 * masses[i] * (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
            for (int d = 0; d < 3; ++d) {
                P[d] += masses[i] * v[d];
            }
        }

        // 4) Potential energy via PM
        auto [phi_flat, weights_list]
            = compute_phi(positions, masses, N, box_size, dp, solver, G_k);

        double PE = 0.0;
        for (size_t i = 0; i < M; ++i) {
            double phi_p = 0.0;
            for (auto& [idx, w] : weights_list[i]) {
                int flat = flatten_index(idx, N);
                phi_p += w * phi_flat[flat];
            }
            PE += masses[i] * phi_p;
        }
        PE *= 0.5;  // account for double‐counting

        double total = KE + PE;

        // 5) Log to console & CSV
        std::cout << "Step " << step
                  << "  KE="    << KE
                  << ", PE="   << PE
                  << ", Total="<< total
                  << ", Px="   << P[0]
                  << ", Py="   << P[1]
                  << ", Pz="   << P[2]
                  << "\n";

        log_file << step << "," << KE << "," << PE << "," << total
                 << "," << P[0] << "," << P[1] << "," << P[2] << "\n";

        // 6) Dump surviving particle positions
        for (size_t i = 0; i < M; ++i) {
            pos_file << step << "," << i << ","
                     << positions[i][0] << ","
                     << positions[i][1] << ","
                     << positions[i][2] << "\n";
        }

        // 7) Record the mid‐plane potential slice
        std::ofstream pot_file("potential_N128.csv", std::ios::app);
        int z = N/2;
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                int flat_idx = x*N*N + y*N + z;
                pot_file << phi_flat[flat_idx] << " ";
            }
            pot_file << "\n";
        }
    }

    std::cout << "dp=" << dp
            << ", threads=" << omp_get_max_threads()
            << ", total_time=" << total_integrator_time
            << ", avg_per_step=" << (total_integrator_time / n_steps)
            << "\n";

    log_file.close();
    pos_file.close();
    return 0;
}
