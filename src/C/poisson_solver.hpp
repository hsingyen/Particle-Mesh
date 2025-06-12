// poisson_solver.hpp
#pragma once

#include <vector>
#include <complex>

/**
 * Solve Poisson equation with periodic boundary conditions using FFT.
 * @param rho      Input density array (flattened size N^3).
 * @param phi      Output potential array (flattened size N^3), pre-sized.
 * @param N        Grid dimension (assume cube N x N x N).
 * @param box_size Physical box length.
 * @param G        Gravitational constant (default 1.0).
 */
void poisson_solver_periodic(const std::vector<double>& rho,
                             std::vector<double>& phi,
                             int N,
                             double box_size,
                             double G = 1.0);

/**
 * Build Green's function in Fourier space for isolated Poisson solver.
 * @param N        Grid dimension for original rho (size N x N x N).
 * @param box_size Physical box length.
 * @return         Complex Green's kernel (flattened size (2N)^2 x ((2N)/2+1)).
 */
std::vector<std::complex<double>> green(int N, double box_size);

/**
 * Solve Poisson equation with isolated boundary using precomputed Green's kernel.
 * @param rho      Input density array (flattened size N^3).
 * @param Gk       Green's kernel in Fourier space.
 * @param phi      Output potential array (flattened size N^3), pre-sized.
 * @param N        Grid dimension.
 * @param box_size Physical box length.
 * @param G        Gravitational constant (default 1.0).
 */
void poisson_solver_isolated(const std::vector<double>& rho,
                             const std::vector<std::complex<double>>& Gk,
                             std::vector<double>& phi,
                             int N,
                             double box_size,
                             double G = 1.0);


// poisson_solver.cpp
#include "poisson_solver.hpp"
#include <fftw3.h>
#include <cmath>
#include <algorithm>
#include <mpi.h>
#include <omp.h>

void poisson_solver_periodic(const std::vector<double>& rho,
                             std::vector<double>& phi,
                             int N,
                             double box_size,
                             double G) {
    int real_size = N * N * N;
    int comp_size = N * N * (N/2 + 1);

    double* rho_in = fftw_alloc_real(real_size);
    fftw_complex* rho_k = fftw_alloc_complex(comp_size);
    fftw_complex* phi_k = fftw_alloc_complex(comp_size);
    double* phi_out = fftw_alloc_real(real_size);

    std::copy(rho.begin(), rho.end(), rho_in);

    auto plan_r2c = fftw_plan_dft_r2c_3d(N, N, N, rho_in, rho_k, FFTW_ESTIMATE);
    auto plan_c2r = fftw_plan_dft_c2r_3d(N, N, N, phi_k, phi_out, FFTW_ESTIMATE);
    fftw_execute(plan_r2c);

    std::vector<double> kfreq(N);
    for (int i = 0; i < N; ++i) {
        kfreq[i] = 2.0 * M_PI * ((i <= N/2) ? i : i - N) / box_size;
    }

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k <= N/2; ++k) {
                int idx = (i * N + j) * (N/2 + 1) + k;
                double kv2 = (idx == 0)
                    ? 1.0
                    : (kfreq[i]*kfreq[i] + kfreq[j]*kfreq[j] + kfreq[k]*kfreq[k]);
                double factor = -4.0 * M_PI * G / kv2;
                phi_k[idx][0] = rho_k[idx][0] * factor;
                phi_k[idx][1] = rho_k[idx][1] * factor;
                if (idx == 0) {
                    phi_k[idx][0] = 0.0;
                    phi_k[idx][1] = 0.0;
                }
            }
        }
    }

    fftw_execute(plan_c2r);

    #pragma omp parallel for
    for (int i = 0; i < real_size; ++i) {
        phi[i] = phi_out[i] / static_cast<double>(real_size);
    }

    fftw_destroy_plan(plan_r2c);
    fftw_destroy_plan(plan_c2r);
    fftw_free(rho_in);
    fftw_free(rho_k);
    fftw_free(phi_k);
    fftw_free(phi_out);
}

std::vector<std::complex<double>> green(int N, double box_size) {
    int n2 = 2 * N;
    int real_size = n2 * n2 * n2;
    int comp_size = n2 * n2 * (n2/2 + 1);

    std::vector<double> g(real_size);
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < n2; ++i) {
        for (int j = 0; j < n2; ++j) {
            for (int k = 0; k < n2; ++k) {
                int idx = (i * n2 + j) * n2 + k;
                int ii = std::min(i, n2 - i);
                int jj = std::min(j, n2 - j);
                int kk = std::min(k, n2 - k);
                double r = std::sqrt(ii*ii + jj*jj + kk*kk);
                g[idx] = (r == 0.0 ? 0.0 : -1.0 / r);
            }
        }
    }

    double* g_in = fftw_alloc_real(real_size);
    fftw_complex* Gk_c = fftw_alloc_complex(comp_size);
    auto plan = fftw_plan_dft_r2c_3d(n2, n2, n2, g_in, Gk_c, FFTW_ESTIMATE);
    std::copy(g.begin(), g.end(), g_in);
    fftw_execute(plan);

    std::vector<std::complex<double>> Gk(comp_size);
    for (int i = 0; i < comp_size; ++i) {
        Gk[i] = {Gk_c[i][0], Gk_c[i][1]};
    }

    fftw_destroy_plan(plan);
    fftw_free(g_in);
    fftw_free(Gk_c);
    return Gk;
}

void poisson_solver_isolated(const std::vector<double>& rho,
                             const std::vector<std::complex<double>>& Gk,
                             std::vector<double>& phi,
                             int N,
                             double box_size,
                             double G) {
    int n2 = 2 * N;
    int real_size2 = n2 * n2 * n2;
    int comp_size2 = n2 * n2 * (n2/2 + 1);

    double* rho_in = fftw_alloc_real(real_size2);
    fftw_complex* rho_k = fftw_alloc_complex(comp_size2);
    fftw_complex* phi_k = fftw_alloc_complex(comp_size2);
    double* phi_out = fftw_alloc_real(real_size2);

    std::fill(rho_in, rho_in + real_size2, 0.0);
    #pragma omp parallel for
    for (int i = 0; i < N*N*N; ++i) rho_in[i] = rho[i];

    auto plan_r2c = fftw_plan_dft_r2c_3d(n2, n2, n2, rho_in, rho_k, FFTW_ESTIMATE);
    auto plan_c2r = fftw_plan_dft_c2r_3d(n2, n2, n2, phi_k, phi_out, FFTW_ESTIMATE);

    fftw_execute(plan_r2c);
    #pragma omp parallel for
    for (int i = 0; i < comp_size2; ++i) {
        auto r = std::complex<double>(rho_k[i][0], rho_k[i][1]);
        auto ph = r * Gk[i];
        phi_k[i][0] = ph.real();
        phi_k[i][1] = ph.imag();
    }

    fftw_execute(plan_c2r);
    double norm = static_cast<double>(real_size2) * (box_size / N);
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                phi[(i*N+j)*N+k] = phi_out[(i*n2+j)*n2+k] / norm;

    fftw_destroy_plan(plan_r2c);
    fftw_destroy_plan(plan_c2r);
    fftw_free(rho_in);
    fftw_free(rho_k);
    fftw_free(phi_k);
    fftw_free(phi_out);
}

// Reminder: initialize MPI in main() and set OpenMP threads with omp_set_num_threads().
