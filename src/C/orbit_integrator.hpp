#pragma once

#include <vector>
#include <array>
#include <string>
#include <complex>
#include <utility>
#include "mass_deposition.hpp"
#include "poisson_solver.hpp"


inline int idx3(int i, int j, int k, int N) {
    return (i * N + j) * N + k;
}


struct StepResult {
    std::vector<std::array<double,3>> positions;
    std::vector<std::array<double,3>> velocities;
    std::vector<double> masses;
    std::vector<double> phi;
};

std::array<std::vector<double>,3> compute_grid_acceleration(
    const std::vector<double>& phi,
    int N,
    double box_size
);

std::vector<std::array<double,3>> interpolate_to_particles(
    const std::array<std::vector<double>,3>& grid_field,
    const std::vector<WeightList>& weights_list,
    int N
);

std::pair<std::vector<double>, std::vector<WeightList>> compute_phi(
    const std::vector<std::array<double,3>>& positions,
    const std::vector<double>& masses,
    int N,
    double box_size,
    const std::string& dp,
    const std::string& solver,
    const std::vector<std::complex<double>>& G_k
);

std::pair<std::vector<std::array<double,3>>, std::vector<double>> compute_acceleration(
    const std::vector<std::array<double,3>>& positions,
    const std::vector<double>& masses,
    int N,
    double box_size,
    const std::string& dp,
    const std::string& solver,
    const std::vector<std::complex<double>>& G_k
);

std::pair<std::vector<std::array<double,3>>, std::vector<double>> nbody_compute_acceleration(
    const std::vector<std::array<double,3>>& positions,
    const std::vector<double>& masses,
    int N,
    double box_size
);

StepResult kdk_step(
    const std::vector<std::array<double,3>>& positions,
    const std::vector<std::array<double,3>>& velocities,
    const std::vector<double>& masses,
    double dt,
    int N,
    double box_size,
    const std::string& dp,
    const std::string& solver,
    const std::vector<std::complex<double>>& G_k
);

StepResult dkd_step(
    const std::vector<std::array<double,3>>& positions,
    const std::vector<std::array<double,3>>& velocities,
    const std::vector<double>& masses,
    double dt,
    int N,
    double box_size,
    const std::string& dp,
    const std::string& solver,
    const std::vector<std::complex<double>>& G_k
);

StepResult rk4_step(
    const std::vector<std::array<double,3>>& positions,
    const std::vector<std::array<double,3>>& velocities,
    const std::vector<double>& masses,
    double dt,
    int N,
    double box_size,
    const std::string& dp,
    const std::string& solver,
    const std::vector<std::complex<double>>& G_k,
    double G = 1.0
);
