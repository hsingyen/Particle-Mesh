// integrator.cpp
#include "orbit_integrator.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <omp.h>

std::array<std::vector<double>,3> compute_grid_acceleration(
    const std::vector<double>& phi,
    int N,
    double box_size,
    const std::string& solver
) {
    double dx = box_size / N;
    int size = N * N * N;
    std::array<std::vector<double>,3> grad;
    grad[0].assign(size, 0.0);
    grad[1].assign(size, 0.0);
    grad[2].assign(size, 0.0);

    #pragma omp parallel for collapse(3)
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            for(int k = 0; k < N; ++k) {
                int idx = idx3(i,j,k,N);
                // X-direction
                double gx;
                if(solver == "periodic") {
                    int ip = (i+1) % N;
                    int im = (i-1+N) % N;
                    gx = -(phi[idx3(ip,j,k,N)] - phi[idx3(im,j,k,N)]) / (2 * dx);
                } else {
                    if(i == 0) {
                        gx = -(-3*phi[idx3(0,j,k,N)] + 4*phi[idx3(1,j,k,N)] - phi[idx3(2,j,k,N)]) / (2 * dx);
                    } else if(i == N-1) {
                        gx = -(3*phi[idx3(N-1,j,k,N)] - 4*phi[idx3(N-2,j,k,N)] + phi[idx3(N-3,j,k,N)]) / (2 * dx);
                    } else {
                        gx = -(phi[idx3(i+1,j,k,N)] - phi[idx3(i-1,j,k,N)]) / (2 * dx);
                    }
                }
                grad[0][idx] = gx;
                // Y-direction
                double gy;
                if(solver == "periodic") {
                    int jp = (j+1) % N;
                    int jm = (j-1+N) % N;
                    gy = -(phi[idx3(i,jp,k,N)] - phi[idx3(i,jm,k,N)]) / (2 * dx);
                } else {
                    if(j == 0) {
                        gy = -(-3*phi[idx3(i,0,k,N)] + 4*phi[idx3(i,1,k,N)] - phi[idx3(i,2,k,N)]) / (2 * dx);
                    } else if(j == N-1) {
                        gy = -(3*phi[idx3(i,N-1,k,N)] - 4*phi[idx3(i,N-2,k,N)] + phi[idx3(i,N-3,k,N)]) / (2 * dx);
                    } else {
                        gy = -(phi[idx3(i,j+1,k,N)] - phi[idx3(i,j-1,k,N)]) / (2 * dx);
                    }
                }
                grad[1][idx] = gy;
                // Z-direction
                double gz;
                if(solver == "periodic") {
                    int kp = (k+1) % N;
                    int km = (k-1+N) % N;
                    gz = -(phi[idx3(i,j,kp,N)] - phi[idx3(i,j,km,N)]) / (2 * dx);
                } else {
                    if(k == 0) {
                        gz = -(-3*phi[idx3(i,j,0,N)] + 4*phi[idx3(i,j,1,N)] - phi[idx3(i,j,2,N)]) / (2 * dx);
                    } else if(k == N-1) {
                        gz = -(3*phi[idx3(i,j,N-1,N)] - 4*phi[idx3(i,j,N-2,N)] + phi[idx3(i,j,N-3,N)]) / (2 * dx);
                    } else {
                        gz = -(phi[idx3(i,j,k+1,N)] - phi[idx3(i,j,k-1,N)]) / (2 * dx);
                    }
                }
                grad[2][idx] = gz;
            }
        }
    }
    return grad;
}

std::vector<std::array<double,3>> interpolate_to_particles(
    const std::array<std::vector<double>,3>& grid_field,
    const std::vector<WeightList>& weights_list,
    int N
) {
    size_t M = weights_list.size();
    std::vector<std::array<double,3>> out(M);

    #pragma omp parallel for
    for(size_t p = 0; p < M; ++p) {
        std::array<double,3> acc = {0,0,0};
        for(auto& w : weights_list[p]) {
            int idx = idx3(w.first.x, w.first.y, w.first.z, N);
            for(int d=0; d<3; ++d)
                acc[d] += grid_field[d][idx] * w.second;
        }
        out[p] = acc;
    }
    return out;
}

std::pair<std::vector<double>, std::vector<WeightList>> compute_phi(
    const std::vector<std::array<double,3>>& positions,
    const std::vector<double>& masses,
    int N,
    double box_size,
    const std::string& dp,
    const std::string& solver,
    const std::vector<std::complex<double>>& G_k
) {
    GridDepositResult dep;
    if(dp=="ngp") dep = deposit_ngp(positions, masses, N, box_size, solver);
    else if(dp=="cic") dep = deposit_cic(positions, masses, N, box_size, solver);
    else if(dp=="tsc") dep = deposit_tsc(positions, masses, N, box_size, solver);
    else throw std::runtime_error("Unknown deposition method");

    std::vector<double> phi;
    if(solver=="periodic") phi = poisson_solver_periodic(dep.rho, N, box_size);
    else phi = poisson_solver_isolated(dep.rho, G_k, N, box_size);
    return {phi, dep.weights_list};
}

std::pair<std::vector<std::array<double,3>>, std::vector<double>> compute_acceleration(
    const std::vector<std::array<double,3>>& positions,
    const std::vector<double>& masses,
    int N,
    double box_size,
    const std::string& dp,
    const std::string& solver,
    const std::vector<std::complex<double>>& G_k
) {
    auto [phi, weights] = compute_phi(positions, masses, N, box_size, dp, solver, G_k);
    auto grid_acc = compute_grid_acceleration(phi, N, box_size, solver);
    auto part_acc = interpolate_to_particles(grid_acc, weights, N);
    return {part_acc, phi};
}

std::pair<std::vector<std::array<double,3>>, std::vector<double>> nbody_compute_acceleration(
    const std::vector<std::array<double,3>>& positions,
    const std::vector<double>& masses,
    int N,
    double box_size
) {
    size_t M = masses.size();
    std::vector<std::array<double,3>> acc(M,{0,0,0});
    double eps = (box_size/(double)N)*(box_size/(double)N);

    #pragma omp parallel for collapse(2)
    for(size_t i=0; i<M; ++i)
    for(size_t j=0; j<M; ++j) {
        if(i==j) continue;
        std::array<double,3> dx;
        for(int d=0; d<3; ++d) dx[d]=positions[i][d]-positions[j][d];
        double r2 = dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2]+eps;
        double invr3 = 1.0/(r2*std::sqrt(r2));
        for(int d=0; d<3; ++d) acc[i][d] -= masses[j]*dx[d]*invr3;
    }
    std::vector<double> phi(N*N*N,0.0);
    #pragma omp parallel for collapse(3)
    for(int i=0;i<N;++i)
    for(int j=0;j<N;++j)
    for(int k=0;k<N;++k) {
        double sum=0;
        double xg=i/(double)N, yg=j/(double)N, zg=k/(double)N;
        for(size_t l=0; l<M; ++l) {
            std::array<double,3> dx={positions[l][0]-xg,positions[l][1]-yg,positions[l][2]-zg};
            double r = std::sqrt(dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2]+eps);
            sum -= masses[l]/r;
        }
        phi[idx3(i,j,k,N)] = sum;
    }
    return {acc, phi};
}

StepResult kdk_step(
    const std::vector<std::array<double,3>>& positions0,
    const std::vector<std::array<double,3>>& velocities0,
    const std::vector<double>& masses0,
    double dt,
    int N,
    double box_size,
    const std::string& dp,
    const std::string& solver,
    const std::vector<std::complex<double>>& G_k
) {
    std::vector<double> masses = masses0;

    auto [acc1, phi1] = compute_acceleration(positions0,masses,N,box_size,dp,solver,G_k);
    auto vel1 = velocities0;
    size_t M = masses.size();
    #pragma omp parallel for
    for(size_t i=0; i<M; ++i)
        for(int d=0; d<3; ++d)
            vel1[i][d] += 0.5*dt*acc1[i][d];
    auto pos1 = positions0;
    #pragma omp parallel for
    for(size_t i=0; i<M; ++i)
        for(int d=0; d<3; ++d)
            pos1[i][d] += dt*vel1[i][d];
    if(solver=="periodic") {
        #pragma omp parallel for
        for(size_t i=0;i<M;++i)
            for(int d=0;d<3;++d)
                pos1[i][d] = std::fmod(pos1[i][d]+box_size,box_size);
    } else {
        // isolated: mask out-of-bounds particles
        std::vector<bool> keep(M);
        #pragma omp parallel for
        for(size_t i=0;i<M;++i) {
            keep[i] = true;
            for(int d=0;d<3;++d) {
                if(pos1[i][d]<0 || pos1[i][d]>=box_size) {
                    keep[i]=false;
                    break;
                }
            }
        }
        std::vector<std::array<double,3>> p2;
        std::vector<std::array<double,3>> v2;
        std::vector<double> m2;
        p2.reserve(M);
        v2.reserve(M);
        m2.reserve(M);        
        for(size_t i=0;i<M;++i) if(keep[i]) {
            p2.push_back(pos1[i]); v2.push_back(vel1[i]);m2.push_back(masses[i]);
        }
        pos1.swap(p2); vel1.swap(v2); 
        m2.swap(masses);
        M = masses.size();

    }

    auto [acc2, phi2] = compute_acceleration(pos1,masses,N,box_size,dp,solver,G_k);
    auto vel2 = vel1;
    #pragma omp parallel for
    for(size_t i=0; i<M; ++i)
        for(int d=0; d<3; ++d)
            vel2[i][d] += 0.5*dt*acc2[i][d];
    return {pos1, vel2, masses, phi2};
}

StepResult dkd_step(
    const std::vector<std::array<double,3>>& pos0,
    const std::vector<std::array<double,3>>& vel0,
    const std::vector<double>& m0,
    double dt,
    int N,
    double box_size,
    const std::string& dp,
    const std::string& solver,
    const std::vector<std::complex<double>>& G_k
) {
    std::vector<double> masses = m0;
    // First drift (half step)
    auto pos1 = pos0;
    std::vector<std::array<double,3>> vel1 = vel0;
    size_t M = masses.size();
    #pragma omp parallel for
    for(size_t i = 0; i < M; ++i)
        for(int d=0; d<3; ++d)
            pos1[i][d] += 0.5 * dt * vel1[i][d];
    // Boundary handling after half drift
    if(solver == "periodic") {
        #pragma omp parallel for
        for(size_t i=0; i<M; ++i)
            for(int d=0; d<3; ++d)
                pos1[i][d] = std::fmod(pos1[i][d] + box_size, box_size);
    } else {
        std::vector<bool> keep(M);
        #pragma omp parallel for
        for(size_t i=0; i<M; ++i) {
            keep[i] = true;
            for(int d=0; d<3; ++d) {
                if(pos1[i][d] < 0 || pos1[i][d] >= box_size) { keep[i] = false; break; }
            }
        }
        std::vector<std::array<double,3>> p2, v2;
        std::vector<double> m2;
        p2.reserve(M);
        v2.reserve(M);
        m2.reserve(M);        
        for(size_t i=0;i<M;++i) if(keep[i]) {
            p2.push_back(pos1[i]); v2.push_back(vel1[i]); m2.push_back(masses[i]);
        }
        pos1.swap(p2); vel1.swap(v2); // masses will use m2 downstream
        m2.swap(masses);
        M = masses.size();
    }
    // Kick (full step) at pos1
    auto [acc1, phi1] = compute_acceleration(pos1, masses, N, box_size, dp, solver, G_k);
    std::vector<std::array<double,3>> vel2 = vel1;
    #pragma omp parallel for
    for(size_t i=0;i<M;++i)
        for(int d=0; d<3; ++d)
            vel2[i][d] += dt * acc1[i][d];
    // Second drift (half step)
    auto pos2 = pos1;
    #pragma omp parallel for
    for(size_t i=0;i<M;++i)
        for(int d=0; d<3; ++d)
            pos2[i][d] += 0.5 * dt * vel2[i][d];
    // Boundary handling after second drift
    if(solver == "periodic") {
        #pragma omp parallel for
        for(size_t i=0; i<M; ++i)
            for(int d=0; d<3; ++d)
                pos2[i][d] = std::fmod(pos2[i][d] + box_size, box_size);
    } else {
        std::vector<bool> keep2(M);
        #pragma omp parallel for
        for(size_t i=0;i<M;++i) {
            keep2[i] = true;
            for(int d=0; d<3; ++d) {
                if(pos2[i][d] < 0 || pos2[i][d] >= box_size) { keep2[i] = false; break; }
            }
        }
        std::vector<std::array<double,3>> p3, v3;
        std::vector<double> m3;
        p3.reserve(M);
        v3.reserve(M);
        m3.reserve(M);

        for(size_t i=0;i<M;++i) if(keep2[i]) {
            p3.push_back(pos2[i]); v3.push_back(vel2[i]); m3.push_back(masses[i]);
        }
        pos2.swap(p3); vel2.swap(v3);
        m3.swap(masses);
        M = masses.size();
    }
    return {pos2, vel2, masses, phi1};
}

StepResult rk4_step(
    const std::vector<std::array<double,3>>& pos0,
    const std::vector<std::array<double,3>>& vel0,
    const std::vector<double>& m0,
    double dt,
    int N,
    double box_size,
    const std::string& dp,
    const std::string& solver,
    const std::vector<std::complex<double>>& G_k,
    double G
) {
    // Helpers to add and scale vector<array<double,3>>
    auto scale = [&](const std::vector<std::array<double,3>>& a, double s){
        std::vector<std::array<double,3>> r = a;
        for(size_t i = 0; i < r.size(); ++i)
            for(int d = 0; d < 3; ++d)
                r[i][d] *= s;
        return r;
    };
    auto add = [&](const std::vector<std::array<double,3>>& a,
                   const std::vector<std::array<double,3>>& b){
        std::vector<std::array<double,3>> r = a;
        for(size_t i = 0; i < r.size(); ++i)
            for(int d = 0; d < 3; ++d)
                r[i][d] += b[i][d];
        return r;
    };

    // Stage 1
    auto acc1 = compute_acceleration(pos0, m0, N, box_size, dp, solver, G_k).first;
    auto k1v = scale(acc1, dt);
    auto k1x = scale(vel0,  dt);

    // Stage 2
    auto pos2 = add(pos0, scale(k1x, 0.5));
    auto vel2 = add(vel0, scale(k1v, 0.5));
    auto acc2 = compute_acceleration(pos2, m0, N, box_size, dp, solver, G_k).first;
    auto k2v = scale(acc2, dt);
    auto k2x = scale(vel2,  dt);

    // Stage 3
    auto pos3 = add(pos0, scale(k2x, 0.5));
    auto vel3 = add(vel0, scale(k2v, 0.5));
    auto acc3 = compute_acceleration(pos3, m0, N, box_size, dp, solver, G_k).first;
    auto k3v = scale(acc3, dt);
    auto k3x = scale(vel3,  dt);

    // Stage 4
    auto pos4 = add(pos0, k3x);
    auto vel4 = add(vel0, k3v);
    auto acc4 = compute_acceleration(pos4, m0, N, box_size, dp, solver, G_k).first;
    auto k4v = scale(acc4, dt);
    auto k4x = scale(vel4,  dt);

    // Combine increments
    auto sumX = add(add(k1x, scale(add(k2x, k3x), 2.0)), k4x);
    auto sumV = add(add(k1v, scale(add(k2v, k3v), 2.0)), k4v);
    auto pos1 = add(pos0, scale(sumX, 1.0/6.0));
    auto vel1 = add(vel0, scale(sumV, 1.0/6.0));

    // Boundary handling & rebuild mass list
    size_t M = m0.size();
    std::vector<double> m2;
    m2.reserve(M);

    if (solver == "periodic") {
        #pragma omp parallel for
        for(size_t i = 0; i < M; ++i)
            for(int d = 0; d < 3; ++d)
                pos1[i][d] = std::fmod(pos1[i][d] + box_size, box_size);
        m2 = m0;
    } else {
        std::vector<bool> keep(M);
        #pragma omp parallel for
        for(size_t i = 0; i < M; ++i) {
            keep[i] = true;
            for(int d = 0; d < 3; ++d) {
                if (pos1[i][d] < 0 || pos1[i][d] >= box_size) {
                    keep[i] = false;
                    break;
                }
            }
        }
        std::vector<std::array<double,3>> p2, v2;
        p2.reserve(M); v2.reserve(M);
        for(size_t i = 0; i < M; ++i) {
            if (keep[i]) {
                p2.push_back(pos1[i]);
                v2.push_back(vel1[i]);
                m2.push_back(m0[i]);
            }
        }
        pos1.swap(p2);
        vel1.swap(v2);
        M = m2.size();
    }

    // Final acceleration & phi on filtered set
    auto phi_final
        = compute_phi(pos1, m2, N, box_size, dp, solver, G_k).first;

    return { pos1, vel1, m2, phi_final };
}