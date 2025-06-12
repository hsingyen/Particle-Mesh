#pragma once

#include <vector>
#include <array>
#include <string>
#include "mass_deposition.hpp"
#include "poisson_solver.hpp"

using ParticleArray = std::vector<std::array<double,3>>;
using MassArray     = std::vector<double>;
using WeightList    = std::vector<std::pair<IndexTriple,double>>;
using WeightsArray  = std::vector<WeightList>;

// 加速度計算結果：每顆粒子的加速度 + 網格上的位勢
struct AccelResult {
    ParticleArray acc;            // 大小 N_particles × 3
    std::vector<double> phi;      // 展平後的 N^3 位勢網格
};

// 積分步驟結果：更新後的位置、速度、質量，以及位勢
struct StepResult {
    ParticleArray positions;
    ParticleArray velocities;
    MassArray     masses;
    std::vector<double> phi;      // 更新後的位勢
};

/** 計算網格上的加速度（中央差分，週期性邊界） */
std::array<std::vector<double>,3> compute_grid_acceleration(
    const std::vector<double>& phi_flat,
    int N,
    double box_size);

/** 將網格場插值回粒子位置 */
ParticleArray interpolate_to_particles(
    const std::array<std::vector<double>,3>& grid_field,
    const WeightsArray& weights_list,
    int N);

/**  
 * 計算位勢 phi 與權重表  
 * - positions, masses: 粒子位置與質量  
 * - dp: 質量分佈方法 ("ngp"/"cic"/"tsc")  
 * - solver: 邊界條件 ("periodic"/"periodic_safe"/"isolated")  
 */
std::pair<std::vector<double>, WeightsArray> compute_phi(
    const ParticleArray& positions,
    const MassArray&     masses,
    int N,
    double box_size,
    const std::string& dp,
    const std::string& solver,
    double soft_len);

/**  
 * 計算粒子加速度（透過網格方法：質量分佈 → 求解 Poisson → 插值）  
 */
AccelResult compute_acceleration(
    const ParticleArray& positions,
    const MassArray&     masses,
    int N,
    double box_size,
    const std::string& dp,
    const std::string& solver,
    bool subtract_self,
    double soft_len);

/**  
 * 直接 N-body 計算加速度，並同時計算網格上的位勢（比較用）  
 */
AccelResult nbody_compute_acceleration(
    const ParticleArray& positions,
    const MassArray&     masses,
    int N,
    double box_size);

/** Kick-Drift-Kick 積分步驟 */
StepResult kdk_step(
    ParticleArray positions,
    ParticleArray velocities,
    MassArray     masses,
    double dt,
    int N,
    double box_size,
    const std::string& dp,
    const std::string& solver,
    bool subtract_self,
    double soft_len);

/** Drift-Kick-Drift 積分步驟 */
StepResult dkd_step(
    ParticleArray positions,
    ParticleArray velocities,
    MassArray     masses,
    double dt,
    int N,
    double box_size,
    const std::string& dp,
    const std::string& solver,
    bool subtract_self,
    double soft_len);

/** 4th-order Runge-Kutta 積分步驟 */
StepResult rk4_step(
    ParticleArray positions,
    ParticleArray velocities,
    MassArray     masses,
    double dt,
    int N,
    double box_size,
    const std::string& dp,
    const std::string& solver,
    bool subtract_self,
    double soft_len,
    double G = 1.0);
