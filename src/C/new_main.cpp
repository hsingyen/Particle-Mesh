#include "orbit_integrator.hpp"
#include "jeans_initial.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <tuple>
#include <cmath>

int main() {
    // === Simulation parameters ===
    const int N = 16;
    const double box_size = 1.0;
    const int N_particles = 100;
    const double dt = 2e-4;
    const int n_steps = 100;
    const std::string dp = "ngp";
    const std::string solver = "periodic";
    const std::string integrator = "dkd";
    const std::string mode = "stable";
    const double a = 0.05;
    const double G = 1.0;

    // === Initialization ===
    ParticleArray positions, velocities;
    MassArray masses;
    std::tie(positions, velocities, masses) = create_particles_single(N_particles, box_size, a, 1.0, mode, solver, G);
    
    std::cout << "[DEBUG] Number of particles: " << positions.size() << std::endl;

    for (size_t i = 0; i < std::min<size_t>(5, velocities.size()); ++i) {
        const auto& v = velocities[i];
        std::cout << "[DEBUG] v[" << i << "] = ("
                << v[0] << ", " << v[1] << ", " << v[2] << ")\n";

        double v2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
        if (!std::isfinite(v2)) {
            std::cerr << "NaN in velocity at particle " << i << ": v = ("
                    << v[0] << "," << v[1] << "," << v[2] << ")\n";
        }
    }
    // === Initial Energy and Momentum ===
    auto res = compute_acceleration(positions, masses, N, box_size, dp, solver, false, 0.0);
    double KE = 0.0, PE = 0.0;
    for (size_t i = 0; i < N_particles; ++i) {
        KE += 0.5 * masses[i] * (velocities[i][0]*velocities[i][0] + velocities[i][1]*velocities[i][1] + velocities[i][2]*velocities[i][2]);
    }
    // NOTE: Skipping potential energy computation from weights for now 

    std::array<double,3> P_total = {0.0, 0.0, 0.0};
    for (size_t i = 0; i < N_particles; ++i)
        for (int d = 0; d < 3; ++d)
            P_total[d] += masses[i] * velocities[i][d];

    std::cout << "Initial KE = " << KE << "\n";
    std::cout << "Initial Momentum = (" << P_total[0] << ", " << P_total[1] << ", " << P_total[2] << ")\n";

    // === Time evolution ===
    for (int step = 0; step < n_steps; ++step) {
        StepResult step_result;
        if (integrator == "kdk") {
            step_result = kdk_step(positions, velocities, masses, dt, N, box_size, dp, solver, false, 0.0);
        } else if (integrator == "dkd") {
            step_result = dkd_step(positions, velocities, masses, dt, N, box_size, dp, solver, false, 0.0);
        } else if (integrator == "rk4") {
            step_result = rk4_step(positions, velocities, masses, dt, N, box_size, dp, solver, false, 0.0, G);
        }

        positions = std::move(step_result.positions);
        velocities = std::move(step_result.velocities);
        masses = std::move(step_result.masses);

        // Diagnostics
        double KE_step = 0.0;
        for (size_t i = 0; i < N_particles; ++i) {
            KE_step += 0.5 * masses[i] * (velocities[i][0]*velocities[i][0] + velocities[i][1]*velocities[i][1] + velocities[i][2]*velocities[i][2]);
        }
        std::cout << "Step " << step << "\t KE = " << KE_step << "\n";
    }

    return 0;
}