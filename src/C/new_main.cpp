#include "orbit_integrator.hpp"
#include "jeans_initial.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <tuple>
#include <cmath>

int flatten_index(const IndexTriple& idx, int N) {
    return idx.x * N * N + idx.y * N + idx.z;
}

int main() {
    // === Simulation Parameters ===
    const int N = 128;
    const double box_size = 1.0;
    const int N_particles = 100;
    const double dt = 2e-4;
    const int n_steps = 100;
    const std::string dp = "ngp";
    const std::string solver = "periodic";
    const std::string integrator = "dkd";
    const std::string mode = "stable";
    const double a = 0.005;
    const double G = 1.0;

    // === Initialization ===
    ParticleArray positions, velocities;
    MassArray masses;
    std::tie(positions, velocities, masses) = create_particles_single(N_particles, box_size, a, 1.0, mode, solver, G);

    std::ofstream log_file("simulation_output.csv");
    log_file << "step,KE,PE,Total,Px,Py,Pz\n";

    std::ofstream pos_file("particle_positions.csv");
    pos_file << "step,id,x,y,z\n";

    std::ofstream clear_file("potential_N128.csv", std::ios::trunc);
    clear_file.close();

    std::vector<std::complex<double>> G_k;
    if (solver == "isolated") {
        G_k = compute_green_ft(N, box_size);
    }

    for (int step = 0; step < n_steps; ++step) {
        StepResult step_result;
        if (integrator == "kdk") {
            step_result = kdk_step(positions, velocities, masses, dt, N, box_size, dp, solver, G_k);
        } else if (integrator == "dkd") {
            step_result = dkd_step(positions, velocities, masses, dt, N, box_size, dp, solver, G_k);
        } else if (integrator == "rk4") {
            step_result = rk4_step(positions, velocities, masses, dt, N, box_size, dp, solver, G_k, G);
        }

        positions = std::move(step_result.positions);
        velocities = std::move(step_result.velocities);
        masses = std::move(step_result.masses);

        double KE = 0.0, PE = 0.0;
        std::array<double, 3> P = {0.0, 0.0, 0.0};
        for (size_t i = 0; i < N_particles; ++i) {
            const auto& v = velocities[i];
            KE += 0.5 * masses[i] * (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
            for (int d = 0; d < 3; ++d) P[d] += masses[i] * v[d];
        }

        auto [phi_flat, weights_list] = compute_phi(positions, masses, N, box_size, dp, solver, G_k);
        
        for (size_t i = 0; i < N_particles; ++i) {
            double phi_particle = 0.0;
            for (const auto& [idx, weight] : weights_list[i]) {
                int flat_idx = flatten_index(idx, N);
                phi_particle += weight * phi_flat[flat_idx];
            }
            PE += masses[i] * phi_particle;
        }

        PE *= 0.5;
        double total = KE + PE;

        std::cout << "Step " << step << "  KE = " << KE << ", PE = " << PE << ", Total = " << total << "\n";
        log_file << step << "," << KE << "," << PE << "," << total << ","
                 << P[0] << "," << P[1] << "," << P[2] << "\n";

        for (size_t i = 0; i < N_particles; ++i)
            pos_file << step << "," << i << "," << positions[i][0] << "," << positions[i][1] << "," << positions[i][2] << "\n";

        std::ofstream pot_file("potential_N128.csv", std::ios::app);
        int z = N / 2;
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                int flat_idx = x * N * N + y * N + z;
                pot_file << phi_flat[flat_idx] << " ";
            }
        }
        pot_file << "\n";
        pot_file.close();
    }

    log_file.close();
    pos_file.close();
    return 0;
}