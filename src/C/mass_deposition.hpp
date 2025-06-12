#pragma once

#include <vector>
#include <array>
#include <string>

/**
 * A triple of grid indices (i,j,k).
 */
struct IndexTriple {
    int x, y, z;
};

/**
 * WeightList for a single particle: list of (grid index, weight).
 */
using WeightList = std::vector<std::pair<IndexTriple, double>>;

/**
 * Result of mass deposition: density grid and per-particle weights.
 * - rho: flattened 3D array of size N*N*N (row-major: i*N*N + j*N + k)
 * - weights_list: vector of WeightList, one per particle
 */
struct GridDepositResult {
    std::vector<double> rho;
    std::vector<WeightList> weights_list;
};

/**
 * Cloud-In-Cell (CIC) mass deposition.
 * @param positions Vector of particle positions (x,y,z)
 * @param masses    Vector of particle masses
 * @param N         Number of grid cells per dimension
 * @param box_size  Physical size of the cubic box
 * @param boundary  "periodic" or "isolated"
 * @return          GridDepositResult containing rho and weights_list
 */
GridDepositResult deposit_cic(
    const std::vector<std::array<double,3>>& positions,
    const std::vector<double>& masses,
    int N,
    double box_size,
    const std::string& boundary);

/**
 * Nearest-Grid-Point (NGP) mass deposition.
 */
GridDepositResult deposit_ngp(
    const std::vector<std::array<double,3>>& positions,
    const std::vector<double>& masses,
    int N,
    double box_size,
    const std::string& boundary);

/**
 * Triangular-Shaped-Cloud (TSC) mass deposition.
 */
GridDepositResult deposit_tsc(
    const std::vector<std::array<double,3>>& positions,
    const std::vector<double>& masses,
    int N,
    double box_size,
    const std::string& boundary);
