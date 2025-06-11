#include "poisson_solver.hpp"
#include <fftw3.h>
#include <cmath>

std::vector<double> poisson_solver_periodic(
    const std::vector<double>& rho_flat,
    int N,
    double box_size,
    double G)
{
    int n = N;
    int n_complex = n * n * (n/2 + 1);
    
    // Allocate FFT arrays
    double* rho   = fftw_alloc_real(n*n*n);
    fftw_complex* rho_k = fftw_alloc_complex(n_complex);
    fftw_complex* phi_k = fftw_alloc_complex(n_complex);
    double* phi   = fftw_alloc_real(n*n*n);

    // Copy density into FFT input
    std::copy(rho_flat.begin(), rho_flat.end(), rho);

    // Create FFTW plans
    fftw_plan plan_r2c = fftw_plan_dft_r2c_3d(n, n, n, rho, rho_k, FFTW_ESTIMATE);
    fftw_plan plan_c2r = fftw_plan_dft_c2r_3d(n, n, n, phi_k, phi, FFTW_ESTIMATE);

    // Forward FFT
    fftw_execute(plan_r2c);

    // Compute k-grid factor
    double dk = 2.0 * M_PI / box_size;
    for (int i = 0; i < n; ++i) {
        double kx = (i <= n/2 ? i : i-n) * dk;
        for (int j = 0; j < n; ++j) {
            double ky = (j <= n/2 ? j : j-n) * dk;
            for (int k = 0; k <= n/2; ++k) {
                double kz = k * dk;
                int idx = (i*n + j)*(n/2+1) + k;
                if (i==0 && j==0 && k==0) {
                    phi_k[idx][0] = phi_k[idx][1] = 0.0;
                } else {
                    double k2 = kx*kx + ky*ky + kz*kz;
                    double factor = -4.0 * M_PI * G / k2;
                    phi_k[idx][0] = rho_k[idx][0] * factor;
                    phi_k[idx][1] = rho_k[idx][1] * factor;
                }
            }
        }
    }

    // Inverse FFT
    fftw_execute(plan_c2r);

    // Normalize and copy output
    std::vector<double> phi_out(n*n*n);
    double scale = 1.0 / (n*n*n);
    for (int idx = 0; idx < n*n*n; ++idx) {
        phi_out[idx] = phi[idx] * scale;
    }

    // Cleanup
    fftw_destroy_plan(plan_r2c);
    fftw_destroy_plan(plan_c2r);
    fftw_free(rho);
    fftw_free(rho_k);
    fftw_free(phi_k);
    fftw_free(phi);

    return phi_out;
}

std::vector<double> poisson_solver_periodic_safe(
    const std::vector<double>& rho_flat,
    int N,
    double box_size,
    double soft_len,
    double G)
{
    int n = N;
    int n_complex = n * n * (n/2 + 1);
    double* rho   = fftw_alloc_real(n*n*n);
    fftw_complex* rho_k = fftw_alloc_complex(n_complex);
    fftw_complex* phi_k = fftw_alloc_complex(n_complex);
    double* phi   = fftw_alloc_real(n*n*n);

    std::copy(rho_flat.begin(), rho_flat.end(), rho);
    fftw_plan plan_r2c = fftw_plan_dft_r2c_3d(n,n,n, rho, rho_k, FFTW_ESTIMATE);
    fftw_plan plan_c2r = fftw_plan_dft_c2r_3d(n,n,n, phi_k, phi, FFTW_ESTIMATE);

    fftw_execute(plan_r2c);
    double dk = 2.0 * M_PI / box_size;
    for (int i = 0; i < n; ++i) {
        double kx = (i <= n/2 ? i : i-n) * dk;
        for (int j = 0; j < n; ++j) {
            double ky = (j <= n/2 ? j : j-n) * dk;
            for (int k = 0; k <= n/2; ++k) {
                double kz = k * dk;
                int idx = (i*n + j)*(n/2+1) + k;
                if (i==0 && j==0 && k==0) {
                    phi_k[idx][0] = phi_k[idx][1] = 0.0;
                } else {
                    double k2 = kx*kx + ky*ky + kz*kz;
                    double soft = (soft_len>0 ? std::exp(-0.5*soft_len*soft_len*k2) : 1.0);
                    double factor = -4.0 * M_PI * G * soft / k2;
                    phi_k[idx][0] = rho_k[idx][0] * factor;
                    phi_k[idx][1] = rho_k[idx][1] * factor;
                }
            }
        }
    }
    fftw_execute(plan_c2r);
    std::vector<double> phi_out(n*n*n);
    double scale = 1.0/(n*n*n);
    for (int idx=0; idx<n*n*n; ++idx) phi_out[idx] = phi[idx]*scale;

    fftw_destroy_plan(plan_r2c);
    fftw_destroy_plan(plan_c2r);
    fftw_free(rho);
    fftw_free(rho_k);
    fftw_free(phi_k);
    fftw_free(phi);
    return phi_out;
}

std::vector<double> poisson_solver_isolated(
    const std::vector<double>& rho_flat,
    int N,
    double box_size,
    double soft_len,
    double G)
{
    int n = N;
    int n2 = 2*n;
    int n2_complex = n2 * n2 * (n2/2 + 1);
    double* rho_pad = fftw_alloc_real(n2*n2*n2);
    fftw_complex* rho_k  = fftw_alloc_complex(n2_complex);
    fftw_complex* phi_k  = fftw_alloc_complex(n2_complex);
    double* phi_pad      = fftw_alloc_real(n2*n2*n2);

    int start = n/2 - 1;
    std::fill(rho_pad, rho_pad + n2*n2*n2, 0.0);
    for (int i=0; i<n; ++i)
    for (int j=0; j<n; ++j)
    for (int k=0; k<n; ++k) {
        int dst = (i+start)*n2*n2 + (j+start)*n2 + (k+start);
        rho_pad[dst] = rho_flat[i*n*n + j*n + k];
    }

    fftw_plan plan_r2c = fftw_plan_dft_r2c_3d(n2,n2,n2, rho_pad, rho_k, FFTW_ESTIMATE);
    fftw_plan plan_c2r = fftw_plan_dft_c2r_3d(n2,n2,n2, phi_k, phi_pad, FFTW_ESTIMATE);

    fftw_execute(plan_r2c);
    double dk = 2.0 * M_PI / box_size;
    for (int i=0; i<n2; ++i) {
        double kx = (i <= n2/2 ? i : i-n2) * dk;
        for (int j=0; j<n2; ++j) {
            double ky = (j <= n2/2 ? j : j-n2) * dk;
            for (int k=0; k<=n2/2; ++k) {
                int idx = (i*n2 + j)*(n2/2+1) + k;
                if (i==0 && j==0 && k==0) {
                    phi_k[idx][0] = phi_k[idx][1] = 0.0;
                } else {
                    double kz = k * dk;
                    double k2 = kx*kx + ky*ky + kz*kz;
                    double soft = (soft_len>0 ? std::exp(-0.5*soft_len*soft_len*k2) : 1.0);
                    double factor = -4.0*M_PI*G * soft / k2;
                    phi_k[idx][0] = rho_k[idx][0] * factor;
                    phi_k[idx][1] = rho_k[idx][1] * factor;
                }
            }
        }
    }

    fftw_execute(plan_c2r);
    std::vector<double> phi_out(n*n*n);
    double scale = 1.0/(n2*n2*n2);
    for (int i=0; i<n; ++i)
    for (int j=0; j<n; ++j)
    for (int k=0; k<n; ++k) {
        int src = (i+start)*n2*n2 + (j+start)*n2 + (k+start);
        phi_out[i*n*n + j*n + k] = phi_pad[src] * scale;
    }

    fftw_destroy_plan(plan_r2c);
    fftw_destroy_plan(plan_c2r);
    fftw_free(rho_pad);
    fftw_free(rho_k);
    fftw_free(phi_k);
    fftw_free(phi_pad);
    return phi_out;
}
