#include "orbit_integrator.hpp"
// #include <mpi.h>
#include <omp.h>
#include <cmath>
#include <stdexcept>

// Compute grid acceleration via central differences (periodic BC)
std::array<std::vector<double>,3> compute_grid_acceleration(
    const std::vector<double>& phi_flat,
    int N,
    double box_size)
{
    double dx = box_size / N;
    std::array<std::vector<double>,3> grad;
    for (int c=0; c<3; ++c) grad[c].assign(N*N*N, 0.0);

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                int ip = (i+1)%N, im = (i-1+N)%N;
                int jp = (j+1)%N, jm = (j-1+N)%N;
                int kp = (k+1)%N, km = (k-1+N)%N;
                int idx  = i*N*N + j*N + k;
                int idx_ip = ip*N*N + j*N + k;
                int idx_im = im*N*N + j*N + k;
                int idx_jp = i*N*N + jp*N + k;
                int idx_jm = i*N*N + jm*N + k;
                int idx_kp = i*N*N + j*N + kp;
                int idx_km = i*N*N + j*N + km;
                grad[0][idx] = -(phi_flat[idx_ip] - phi_flat[idx_im]) / (2*dx);
                grad[1][idx] = -(phi_flat[idx_jp] - phi_flat[idx_jm]) / (2*dx);
                grad[2][idx] = -(phi_flat[idx_kp] - phi_flat[idx_km]) / (2*dx);
            }
        }
    }
    return grad;
}

// Interpolate grid field back to particles
ParticleArray interpolate_to_particles(
    const std::array<std::vector<double>,3>& grid_field,
    const WeightsArray& weights_list,
    int N)
{
    ParticleArray values(weights_list.size(), {0.0,0.0,0.0});
    #pragma omp parallel for
    for (size_t p = 0; p < weights_list.size(); ++p) {
        std::array<double,3> acc = {0.0,0.0,0.0};
        for (const auto& pr : weights_list[p]) {
            const IndexTriple& I = pr.first;
            int idx = I.x*N*N + I.y*N + I.z;
            double w = pr.second;
            acc[0] += grid_field[0][idx] * w;
            acc[1] += grid_field[1][idx] * w;
            acc[2] += grid_field[2][idx] * w;
        }
        values[p] = acc;
    }
    return values;
}

// Compute potential and weight lists via deposition + Poisson solver
std::pair<std::vector<double>, WeightsArray> compute_phi(
    const ParticleArray& positions,
    const MassArray& masses,
    int N,
    double box_size,
    const std::string& dp,
    const std::string& solver,
    double soft_len)
{
    GridDepositResult dep;
    if (dp == "ngp") dep = deposit_ngp(positions, masses, N, box_size, solver);
    else if (dp == "cic") dep = deposit_cic(positions, masses, N, box_size, solver);
    else if (dp == "tsc") dep = deposit_tsc(positions, masses, N, box_size, solver);
    else throw std::invalid_argument("Unknown deposition method");

    std::vector<double> phi;
    if (solver == "periodic") phi = poisson_solver_periodic(dep.rho, N, box_size);
    else if (solver == "periodic_safe") phi = poisson_solver_periodic_safe(dep.rho, N, box_size, soft_len);
    else if (solver == "isolated") phi = poisson_solver_isolated(dep.rho, N, box_size, soft_len);
    else throw std::invalid_argument("Unknown solver method");

    return {std::move(phi), std::move(dep.weights_list)};
}

// Mesh-based acceleration computation
AccelResult compute_acceleration(
    const ParticleArray& positions,
    const MassArray& masses,
    int N,
    double box_size,
    const std::string& dp,
    const std::string& solver,
    bool /*subtract_self*/, double soft_len)
{
    auto [phi, weights] = compute_phi(positions, masses, N, box_size, dp, solver, soft_len);
    auto grid_acc = compute_grid_acceleration(phi, N, box_size);
    auto acc_part = interpolate_to_particles(grid_acc, weights, N);
    return {std::move(acc_part), std::move(phi)};
}

// Direct N-body acceleration + grid potential
AccelResult nbody_compute_acceleration(
    const ParticleArray& positions,
    const MassArray& masses,
    int N,
    double box_size)
{
    size_t M = masses.size();
    ParticleArray acc(M, {0.0,0.0,0.0});
    double soft = box_size / (N*N);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < M; ++j) {
            if (i == j) continue;
            std::array<double,3> d;
            for (int dind = 0; dind < 3; ++dind)
                d[dind] = positions[i][dind] - positions[j][dind];
            double r2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2] + soft;
            double inv_r3 = 1.0 / std::pow(r2, 1.5);

            double fx = masses[j] * d[0] * inv_r3;
            double fy = masses[j] * d[1] * inv_r3;
            double fz = masses[j] * d[2] * inv_r3;

            #pragma omp atomic
            acc[i][0] -= fx;
            #pragma omp atomic
            acc[i][1] -= fy;
            #pragma omp atomic
            acc[i][2] -= fz;
        }
    }

    std::vector<double> phi(N*N*N, 0.0);
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) for (int k = 0; k < N; ++k) {
        double val = 0.0;
        for (size_t l = 0; l < M; ++l) {
            double dx = positions[l][0] - double(i)/N;
            double dy = positions[l][1] - double(j)/N;
            double dz = positions[l][2] - double(k)/N;
            double r = std::sqrt(dx*dx + dy*dy + dz*dz + soft);
            val -= masses[l] / r;
        }
        phi[i*N*N + j*N + k] = val;
    }
    return {std::move(acc), std::move(phi)};
}

// Kick-Drift-Kick integration
StepResult kdk_step(
    ParticleArray positions,
    ParticleArray velocities,
    MassArray masses,
    double dt,
    int N,
    double box_size,
    const std::string& dp,
    const std::string& solver,
    bool subtract_self,
    double soft_len)
{
    auto res1 = compute_acceleration(positions, masses, N, box_size, dp, solver, subtract_self, soft_len);
    auto& a1 = res1.acc;

    #pragma omp parallel for
    for (size_t p = 0; p < positions.size(); ++p)
        for (int d = 0; d < 3; ++d)
            velocities[p][d] += 0.5 * dt * a1[p][d];

    #pragma omp parallel for
    for (size_t p = 0; p < positions.size(); ++p)
        for (int d = 0; d < 3; ++d)
            positions[p][d] += dt * velocities[p][d];

    if (solver == "periodic") {
        for (auto& pos : positions)
            for (int d = 0; d < 3; ++d)
                pos[d] = fmod(pos[d], box_size);
    }

    auto res2 = compute_acceleration(positions, masses, N, box_size, dp, solver, subtract_self, soft_len);
    auto& a2 = res2.acc;

    #pragma omp parallel for
    for (size_t p = 0; p < positions.size(); ++p)
        for (int d = 0; d < 3; ++d)
            velocities[p][d] += 0.5 * dt * a2[p][d];

    return {std::move(positions), std::move(velocities), std::move(masses), std::move(res2.phi)};
}

// Drift-Kick-Drift integration
StepResult dkd_step(
    ParticleArray positions,
    ParticleArray velocities,
    MassArray masses,
    double dt,
    int N,
    double box_size,
    const std::string& dp,
    const std::string& solver,
    bool subtract_self,
    double soft_len)
{
    #pragma omp parallel for
    for (size_t p = 0; p < positions.size(); ++p)
        for (int d = 0; d < 3; ++d)
            positions[p][d] += 0.5 * dt * velocities[p][d];

    if (solver == "periodic")
        for (auto& pos : positions)
            for (int d = 0; d < 3; ++d)
                pos[d] = fmod(pos[d], box_size);

    auto res = compute_acceleration(positions, masses, N, box_size, dp, solver, subtract_self, soft_len);
    auto& acc = res.acc;

    #pragma omp parallel for
    for (size_t p = 0; p < positions.size(); ++p)
        for (int d = 0; d < 3; ++d)
            velocities[p][d] += dt * acc[p][d];

    #pragma omp parallel for
    for (size_t p = 0; p < positions.size(); ++p)
        for (int d = 0; d < 3; ++d)
            positions[p][d] += 0.5 * dt * velocities[p][d];

    if (solver == "periodic")
        for (auto& pos : positions)
            for (int d = 0; d < 3; ++d)
                pos[d] = fmod(pos[d], box_size);

    return {std::move(positions), std::move(velocities), std::move(masses), std::move(res.phi)};
}

// 4th-order Runge-Kutta integration
StepResult rk4_step(
    ParticleArray positions,
    ParticleArray velocities,
    MassArray masses,
    double dt,
    int N,
    double box_size,
    const std::string& dp,
    const std::string& solver,
    bool subtract_self,
    double soft_len,
    double G)
{
    auto r1 = compute_acceleration(positions, masses, N, box_size, dp, solver, subtract_self, soft_len);
    ParticleArray k1x = velocities;
    ParticleArray k1v = r1.acc;
    for (auto& v : k1v) for(double& c : v) c *= dt;
    for (auto& x : k1x) for(double& c : x) c *= dt;

    ParticleArray pos2 = positions;
    ParticleArray vel2 = velocities;
    for (size_t p = 0; p < positions.size(); ++p) {
        for (int d = 0; d < 3; ++d) {
            pos2[p][d] += 0.5 * k1x[p][d];
            vel2[p][d] += 0.5 * k1v[p][d];
        }
    }
    auto r2 = compute_acceleration(pos2, masses, N, box_size, dp, solver, subtract_self, soft_len);
    ParticleArray k2x = vel2;
    ParticleArray k2v = r2.acc;
    for (auto& v : k2v) for(double& c : v) c *= dt;
    for (auto& x : k2x) for(double& c : x) c *= dt;

    ParticleArray pos3 = positions;
    ParticleArray vel3 = velocities;
    for (size_t p = 0; p < positions.size(); ++p) {
        for (int d = 0; d < 3; ++d) {
            pos3[p][d] += 0.5 * k2x[p][d];
            vel3[p][d] += 0.5 * k2v[p][d];
        }
    }
    auto r3 = compute_acceleration(pos3, masses, N, box_size, dp, solver, subtract_self, soft_len);
    ParticleArray k3x = vel3;
    ParticleArray k3v = r3.acc;
    for (auto& v : k3v) for(double& c : v) c *= dt;
    for (auto& x : k3x) for(double& c : x) c *= dt;

    ParticleArray pos4 = positions;
    ParticleArray vel4 = velocities;
    for (size_t p = 0; p < positions.size(); ++p) {
        for (int d = 0; d < 3; ++d) {
            pos4[p][d] += k3x[p][d];
            vel4[p][d] += k3v[p][d];
        }
    }
    auto r4 = compute_acceleration(pos4, masses, N, box_size, dp, solver, subtract_self, soft_len);
    ParticleArray k4x = vel4;
    ParticleArray k4v = r4.acc;
    for (auto& v : k4v) for(double& c : v) c *= dt;
    for (auto& x : k4x) for(double& c : x) c *= dt;

    for (size_t p = 0; p < positions.size(); ++p) {
        for (int d = 0; d < 3; ++d) {
            positions[p][d] += (k1x[p][d] + 2*k2x[p][d] + 2*k3x[p][d] + k4x[p][d]) / 6.0;
            velocities[p][d] += (k1v[p][d] + 2*k2v[p][d] + 2*k3v[p][d] + k4v[p][d]) / 6.0;
        }
    }

    if (solver == "periodic") {
        for (auto& pos : positions)
            for (int d = 0; d < 3; ++d)
                pos[d] = fmod(pos[d], box_size);
    }

    return {std::move(positions), std::move(velocities), std::move(masses), std::move(r4.phi)};
}