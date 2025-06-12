#include "jeans_initial.hpp"
#include <random>
#include <cmath>
#include <vector>
#include <array>
#include <string>
#include <stdexcept>
#include <iostream>
#include <ostream>
#include <algorithm>

using ParticleArray = std::vector<std::array<double, 3>>;
using MassArray = std::vector<double>;

namespace {
    double plummer_velocity_dispersion(double r, double a, double M, double G = 1.0) {
        return std::sqrt(G * M / std::sqrt(r*r + a*a) / 6.0);
    }
}

ParticleArray create_plummer_positions(int N, double box_size, double a, double shift) {
    ParticleArray positions(N);
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    for (int i = 0; i < N; ++i) {
        double u = uni(gen);
        u = std::clamp(u, 1e-6, 1.0 - 1e-6);  // avoid u = 0 or 1

        double temp = std::pow(u, -2.0/3.0) - 1.0;
        double r = a / std::sqrt(temp);     // Plummer radius


        // Sample direction uniformly on the sphere
        double theta = std::acos(1 - 2 * uni(gen));
        double phi = 2 * M_PI * uni(gen);

        // Now convert (r, θ, φ) to Cartesian if needed
        double x = r * std::sin(theta) * std::cos(phi);
        double y = r * std::sin(theta) * std::sin(phi);
        double z = r * std::cos(theta);

        positions[i] = {x + box_size/2 + shift, y + box_size/2, z + box_size/2};
    }
    return positions;
}

ParticleArray create_plummer_velocities(
    const std::vector<double>& r, double a, double M, const std::string& mode, bool add_bulk, double bulk_v)
{
    int N = r.size();
    ParticleArray velocities(N);
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> normal(0.0, 1.0);

    double scale = (mode == "stable" ? 1.0 : (mode == "contract" ? 0.5 : (mode == "expand" ? 1.5 : -1.0)));
    if (scale < 0.0) throw std::invalid_argument("Unknown mode: " + mode);

    for (int i = 0; i < N; ++i) {
        double sigma = plummer_velocity_dispersion(r[i], a, M);
        for (int d = 0; d < 3; ++d) {
            velocities[i][d] = normal(gen) * sigma * scale;
        }
        if (add_bulk) velocities[i][0] += bulk_v;
    }
    return velocities;
}

std::tuple<ParticleArray, ParticleArray, MassArray> create_particles_single(
    int N_particles, double box_size, double a, double M,
    const std::string& mode, const std::string& solver, double G)
{
    ParticleArray pos = create_plummer_positions(N_particles, box_size, a, 0.0);

    std::vector<double> radii(N_particles);
    for (int i = 0; i < N_particles; ++i) {
        auto& p = pos[i];
        double dx = p[0] - box_size/2;
        double dy = p[1] - box_size/2;
        double dz = p[2] - box_size/2;
        radii[i] = std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    ParticleArray vel = create_plummer_velocities(radii, a, M, mode, false, 0.0);
    MassArray mass(N_particles, M / N_particles);

    if (solver == "isolated") {
        ParticleArray p_new, v_new;
        MassArray m_new;
        for (int i = 0; i < N_particles; ++i) {
            const auto& p = pos[i];
            if ((p[0]>=0 && p[0]<box_size) && (p[1]>=0 && p[1]<box_size) && (p[2]>=0 && p[2]<box_size)) {
                p_new.push_back(p);
                v_new.push_back(vel[i]);
                m_new.push_back(mass[i]);
            }
        }
        return {p_new, v_new, m_new};
    }
    return {pos, vel, mass};
}

std::tuple<ParticleArray, ParticleArray, MassArray> create_particles_double(
    int N_particles, double box_size, double a, double M,
    const std::string& mode, const std::string& solver, double G,
    bool add_initial_velocity, double v_offset)
{
    int N_each = N_particles / 2;
    ParticleArray pos_all, vel_all;
    MassArray mass(N_particles, M / N_particles);

    for (auto [shift, bulk_v] : std::vector<std::pair<double, double>>{{-0.1, +v_offset}, {+0.1, -v_offset}}) {
        ParticleArray pos = create_plummer_positions(N_each, box_size, a, shift);

        std::vector<double> radii(N_each);
        for (int i = 0; i < N_each; ++i) {
            auto& p = pos[i];
            double dx = p[0] - box_size/2;
            double dy = p[1] - box_size/2;
            double dz = p[2] - box_size/2;
            radii[i] = std::sqrt(dx*dx + dy*dy + dz*dz);
        }
        ParticleArray vel = create_plummer_velocities(radii, a, M, mode, add_initial_velocity, bulk_v);
        pos_all.insert(pos_all.end(), pos.begin(), pos.end());
        vel_all.insert(vel_all.end(), vel.begin(), vel.end());
    }

    if (solver == "isolated") {
        ParticleArray p_new, v_new;
        MassArray m_new;
        for (size_t i = 0; i < pos_all.size(); ++i) {
            const auto& p = pos_all[i];
            if ((p[0]>=0 && p[0]<box_size) && (p[1]>=0 && p[1]<box_size) && (p[2]>=0 && p[2]<box_size)) {
                p_new.push_back(p);
                v_new.push_back(vel_all[i]);
                m_new.push_back(mass[i]);
            }
        }
        return {p_new, v_new, m_new};
    }
    return {pos_all, vel_all, mass};
}