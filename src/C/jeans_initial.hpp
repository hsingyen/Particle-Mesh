#pragma once
#include <vector>
#include <array>
#include <string>
#include <tuple>

using ParticleArray = std::vector<std::array<double, 3>>;
using MassArray = std::vector<double>;

ParticleArray create_plummer_positions(int N, double box_size, double a, double shift = 0.0);

ParticleArray create_plummer_velocities(
    const std::vector<double>& r,
    double a,
    double M,
    const std::string& mode,
    bool add_bulk,
    double bulk_v = 0.0);

std::tuple<ParticleArray, ParticleArray, MassArray> create_particles_single(
    int N_particles,
    double box_size,
    double a,
    double M,
    const std::string& mode,
    const std::string& solver,
    double G);

std::tuple<ParticleArray, ParticleArray, MassArray> create_particles_double(
    int N_particles,
    double box_size,
    double a,
    double M,
    const std::string& mode,
    const std::string& solver,
    double G,
    bool add_initial_velocity,
    double v_offset);