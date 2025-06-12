// poisson_solver.cpp
#include "poisson_solver.hpp"
#include <fftw3.h>
#include <cmath>
#include <omp.h>

// Helper to flatten 3D index
inline int idx3(int i, int j, int k, int N) {
    return (i * N + j) * N + k;
}

std::vector<double> poisson_solver_periodic(
    const std::vector<double>& rho,
    int N,
    double box_size,
    double G
) {
    int size = N*N*N;
    // 1) copy density into FFTW input
    std::vector<double> in(rho);
    std::vector<std::complex<double>> out(size);

    // 2) forward FFT
    fftw_plan plan_f = fftw_plan_dft_r2c_3d(
        N, N, N,
        in.data(),
        reinterpret_cast<fftw_complex*>(out.data()),
        FFTW_ESTIMATE
    );
    fftw_execute(plan_f);
    fftw_destroy_plan(plan_f);

    // 3) build k^2 and divide
    //double factor = 2.0 * M_PI / (box_size/N);
    double factor = 2.0 * M_PI / (box_size);

    // note: ffwt_r2c packs k-space as [0..N-1][0..N-1][0..N/2]
    int nkz = N/2 + 1;
#pragma omp parallel for collapse(3)
    for(int i=0;i<N;i++){
      for(int j=0;j<N;j++){
        for(int k=0;k<nkz;k++){
          int index = (i*N + j)*nkz + k;
          double kx = factor * ((i<=N/2)? i : i-N);
          double ky = factor * ((j<=N/2)? j : j-N);
          double kz = factor * k;
          double k2 = kx*kx + ky*ky + kz*kz;
          if(k==0 && i==0 && j==0){
            out[index] = 0.0;
          } else {
            out[index] *= (-4.0 * M_PI * G) / k2;
          }
        }
      }
    }

    // 4) inverse FFT
    std::vector<double> phi(size);
    fftw_plan plan_b = fftw_plan_dft_c2r_3d(
        N, N, N,
        reinterpret_cast<fftw_complex*>(out.data()),
        phi.data(),
        FFTW_ESTIMATE
    );
    fftw_execute(plan_b);
    fftw_destroy_plan(plan_b);

    // 5) normalize (FFTW does unnormalized transforms)
    double norm = 1.0/size;
#pragma omp parallel for
    for(int i=0;i<size;i++){
      phi[i] *= norm;
    }

    return phi;
}

std::vector<double> poisson_solver_isolated(
    const std::vector<double>& rho,
    const std::vector<std::complex<double>>& G_k,
    int N,
    double box_size,
    double G
) {
    int N2 = 2*N;
    int size_pad = N2*N2*N2;
    int nkz = N2/2 + 1;

    // 1) pad rho
    std::vector<double> in(size_pad, 0.0);
#pragma omp parallel for collapse(3)
    for(int i=0;i<N;i++){
      for(int j=0;j<N;j++){
        for(int k=0;k<N;k++){
          in[idx3(i,j,k,N2)] = rho[idx3(i,j,k,N)];
        }
      }
    }

    // 2) forward FFT
    std::vector<std::complex<double>> rho_k(size_pad);
    fftw_plan plan_f = fftw_plan_dft_r2c_3d(
        N2, N2, N2,
        in.data(),
        reinterpret_cast<fftw_complex*>(rho_k.data()),
        FFTW_ESTIMATE
    );
    fftw_execute(plan_f);
    fftw_destroy_plan(plan_f);

    // 3) multiply by Green’s function in k-space
#pragma omp parallel for
    for(int idx=0; idx < (int)rho_k.size(); idx++){
      rho_k[idx] *= G_k[idx];
    }

    // 4) inverse FFT
    std::vector<double> phi_pad(size_pad);
    fftw_plan plan_b = fftw_plan_dft_c2r_3d(
        N2, N2, N2,
        reinterpret_cast<fftw_complex*>(rho_k.data()),
        phi_pad.data(),
        FFTW_ESTIMATE
    );
    fftw_execute(plan_b);
    fftw_destroy_plan(plan_b);

    // 5) normalize and extract
    double dx = box_size / N;
    double norm = 1.0/(dx*dx)/size_pad;  // account for volume element and FFT scaling
#pragma omp parallel for
    for(int i=0;i<size_pad;i++){
      phi_pad[i] *= norm * (4.0*M_PI*G);
    }

    std::vector<double> phi(N*N*N);
#pragma omp parallel for collapse(3)
    for(int i=0;i<N;i++){
      for(int j=0;j<N;j++){
        for(int k=0;k<N;k++){
          phi[idx3(i,j,k,N)] = phi_pad[idx3(i,j,k,N2)];
        }
      }
    }

    return phi;
}

std::vector<std::complex<double>> compute_green_ft(
    int N,
    double box_size
) {
    int N2 = 2*N;
    int size = N2*N2*N2;
    std::vector<double> g(size);
    double dx = box_size / N;

    // 1) fill real-space Green’s function
#pragma omp parallel for collapse(3)
    for(int i=0; i < N2; i++){
      for(int j=0; j < N2; j++){
        for(int k=0; k < N2; k++){
          int ii = std::min(i, N2 - i);
          int jj = std::min(j, N2 - j);
          int kk = std::min(k, N2 - k);
          double r = std::sqrt(ii*ii + jj*jj + kk*kk);
          if(r == 0) g[idx3(i,j,k,N2)] = 0.0;
          else       g[idx3(i,j,k,N2)] = -1.0 / r;
        }
      }
    }

    // 2) FFT to get G_k
    std::vector<std::complex<double>> G_k(size);
    fftw_plan plan_f = fftw_plan_dft_r2c_3d(
        N2, N2, N2,
        g.data(),
        reinterpret_cast<fftw_complex*>(G_k.data()),
        FFTW_ESTIMATE
    );
    fftw_execute(plan_f);
    fftw_destroy_plan(plan_f);

    return G_k;
} 