#include "mass_deposition.hpp"
// #include <mpi.h>
#include <omp.h>
#include <cmath>

GridDepositResult deposit_cic(
    const std::vector<std::array<double,3>>& positions,
    const std::vector<double>& masses,
    int N,
    double box_size,
    const std::string& boundary)
{
    double dx = box_size / static_cast<double>(N);
    std::vector<double> rho(N*N*N, 0.0);
    std::vector<WeightList> weights_list(positions.size());

    #pragma omp parallel for schedule(static)
    for (size_t p = 0; p < positions.size(); ++p) {
        const auto& pos = positions[p];
        double m = masses[p];
        double xg = pos[0] / dx;
        double yg = pos[1] / dx;
        double zg = pos[2] / dx;

        int ix = static_cast<int>(std::floor(xg-0.5));
        int iy = static_cast<int>(std::floor(yg-0.5));
        int iz = static_cast<int>(std::floor(zg-0.5));

        double dx1 = xg -0.5 - ix;
        double dy1 = yg -0.5 - iy;
        double dz1 = zg -0.5 - iz;
 
        double w[8] = {
            (1-dx1)*(1-dy1)*(1-dz1), dx1*(1-dy1)*(1-dz1),
            (1-dx1)*dy1*(1-dz1), (1-dx1)*(1-dy1)*dz1,
            dx1*dy1*(1-dz1), dx1*(1-dy1)*dz1,
            (1-dx1)*dy1*dz1, dx1*dy1*dz1
        };

        struct IndexTriple neigh[8] = {{ix,iy,iz},{ix+1,iy,iz},{ix,iy+1,iz},{ix,iy,iz+1},
                                       {ix+1,iy+1,iz},{ix+1,iy,iz+1},{ix,iy+1,iz+1},{ix+1,iy+1,iz+1}};

        WeightList local;
        local.reserve(8);

        for (int k = 0; k < 8; ++k) {
            int i = neigh[k].x;
            int j = neigh[k].y;
            int l = neigh[k].z;
            double weight = w[k];

            if (boundary == "periodic") {
                i = (i % N + N) % N;
                j = (j % N + N) % N;
                l = (l % N + N) % N;
            }
            if (boundary == "periodic" ||
                (boundary == "isolated" && i>=0 && i<N && j>=0 && j<N && l>=0 && l<N)) {
                int idx = i * N * N + j * N + l;
                double delta = m * weight / (dx*dx*dx);
                #pragma omp atomic
                rho[idx] += delta;
                local.emplace_back(IndexTriple{i,j,l}, weight);
            }
        }
        weights_list[p] = std::move(local);
    }
    return GridDepositResult{std::move(rho), std::move(weights_list)};
}

GridDepositResult deposit_ngp(
    const std::vector<std::array<double,3>>& positions,
    const std::vector<double>& masses,
    int N,
    double box_size,
    const std::string& boundary)
{
    double dx = box_size / static_cast<double>(N);
    double inv_vol = 1.0/(dx*dx*dx);
    std::vector<double> rho(N*N*N, 0.0);
    std::vector<WeightList> weights_list(positions.size());

    #pragma omp parallel for schedule(static)
    for (size_t p = 0; p < positions.size(); ++p) {
        const auto& pos = positions[p];
        double m = masses[p];
        int ix = static_cast<int>(std::floor(pos[0] / dx));
        int iy = static_cast<int>(std::floor(pos[1] / dx));
        int iz = static_cast<int>(std::floor(pos[2] / dx));
        if (boundary == "periodic") {
            ix = (ix % N + N) % N;
            iy = (iy % N + N) % N;
            iz = (iz % N + N) % N;
        }
        bool in_bounds = (ix>=0 && ix<N && iy>=0 && iy<N && iz>=0 && iz<N);
        WeightList local;
        local.reserve(1);
        if (boundary == "periodic" || (boundary == "isolated" && in_bounds)) {
            int idx = ix * N * N + iy * N + iz;
            #pragma omp atomic
            rho[idx] += m * inv_vol;
            local.emplace_back(IndexTriple{ix,iy,iz}, 1.0);
        }
        weights_list[p] = std::move(local);
    }
    return GridDepositResult{std::move(rho), std::move(weights_list)};
}

GridDepositResult deposit_tsc(
    const std::vector<std::array<double,3>>& positions,
    const std::vector<double>& masses,
    int N,
    double box_size,
    const std::string& boundary)
{
    double dx = box_size / static_cast<double>(N);
    double inv_vol = 1.0/(dx*dx*dx);
    auto tsc_w = [&](double r) {
        double ar = std::abs(r);
        if (ar < 0.5)       return 0.75 - ar*ar;
        else if (ar < 1.5)  return 0.5*(1.5-ar)*(1.5-ar);
        else                return 0.0;
    };
    std::vector<double> rho(N*N*N, 0.0);
    std::vector<WeightList> weights_list(positions.size());

    #pragma omp parallel for schedule(static)
    for (size_t p = 0; p < positions.size(); ++p) {
        auto pos = positions[p];
        double m = masses[p];
        if (boundary == "periodic") {
            for (int d=0; d<3; ++d) {
                pos[d] = fmod(pos[d], box_size);
                if (pos[d]<0) pos[d] += box_size;
            }
        }
        double xg = pos[0]/dx, yg = pos[1]/dx, zg = pos[2]/dx;
        int ix = static_cast<int>(std::floor(xg));
        int iy = static_cast<int>(std::floor(yg));
        int iz = static_cast<int>(std::floor(zg));
        WeightList local;
        local.reserve(27);
        for (int dx_idx=-1; dx_idx<=1; ++dx_idx)
        for (int dy_idx=-1; dy_idx<=1; ++dy_idx)
        for (int dz_idx=-1; dz_idx<=1; ++dz_idx) {
            int i = ix+dx_idx, j = iy+dy_idx, k = iz+dz_idx;
            double w = tsc_w(xg -0.5 - (ix+dx_idx)) * tsc_w(yg -0.5 - (iy+dy_idx)) * tsc_w(zg -0.5 - (iz+dz_idx));
            if (w == 0.0) continue;
            if (boundary == "periodic") {
                i = (i % N + N) % N;
                j = (j % N + N) % N;
                k = (k % N + N) % N;
            } else if (!(i>=0&&i<N && j>=0&&j<N && k>=0&&k<N)) {
                continue;
            }
            int idx = i * N * N + j * N + k;
            #pragma omp atomic
            rho[idx] += m * w * inv_vol;
            local.emplace_back(IndexTriple{i,j,k}, w);
        }
        weights_list[p] = std::move(local);
    }
    return GridDepositResult{std::move(rho), std::move(weights_list)};
}
