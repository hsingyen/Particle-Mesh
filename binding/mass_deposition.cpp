#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <omp.h>

namespace py = pybind11;
std::pair< py::array_t<double>, py::list > deposit_ngp(
    py::array_t<double, py::array::c_style> positions,
    py::array_t<double, py::array::c_style> masses,
    int N,
    double box_size,
    const std::string &boundary)
{
    omp_set_num_threads(6);
    
    auto buf_pos = positions.request();
    if (buf_pos.ndim != 2 || buf_pos.shape[1] != 3) {
        throw std::runtime_error("positions must be shape=(N,3) float64 C-contiguous ndarray");
    }1
    auto buf_mass = masses.request();
    if (buf_mass.ndim !=  || buf_mass.shape[0] != buf_pos.shape[0]) {
        throw std::runtime_error("masses must be shape=(N,) float64 C-contiguous ndarray");
    }

    double *pos_ptr  = static_cast<double*>(buf_pos.ptr);
    double *mass_ptr = static_cast<double*>(buf_mass.ptr);
    ssize_t M = buf_pos.shape[0];      
    ssize_t cols = buf_pos.shape[1];   

    py::array_t<double> rho({N, N, N});
    auto buf_rho = rho.request();
    double *rho_ptr = static_cast<double*>(buf_rho.ptr);
    ssize_t total_rho = N * N * N;
    for (ssize_t i = 0; i < total_rho; ++i) {
        rho_ptr[i] = 0.0;
    }

    double dx     = box_size / static_cast<double>(N);
    double inv_vol = 1.0 / (dx * dx * dx);
    int NN = N * N;
    // pre-allocate weight index storage
    std::vector<std::array<int,3>> weight_idx(M);
    

    // parallel loop over particles
    #pragma omp parallel for
    for (ssize_t i = 0; i < M; ++i) {
        if(i == 0)
            std::cout<<"Total threads: "<<omp_get_num_threads();

        double x = pos_ptr[i * cols + 0] / dx;
        double y = pos_ptr[i * cols + 1] / dx;
        double z = pos_ptr[i * cols + 2] / dx;

        int ix = static_cast<int>(std::round(x));
        int iy = static_cast<int>(std::round(y));
        int iz = static_cast<int>(std::round(z));

        if (boundary == "periodic") {
            ix %= N; if (ix < 0) ix += N;
            iy %= N; if (iy < 0) iy += N;
            iz %= N; if (iz < 0) iz += N;
            #pragma omp atomic
            rho_ptr[ static_cast<ssize_t>(ix)*NN + static_cast<ssize_t>(iy)*N + iz ] += mass_ptr[i] * inv_vol;
            weight_idx[i] = {ix, iy, iz};

        }
        else if (boundary == "isolated") {
            if (0 <= ix && ix < N && 0 <= iy && iy < N && 0 <= iz && iz < N) {
                #pragma omp atomic
                rho_ptr[ static_cast<ssize_t>(ix)*NN + static_cast<ssize_t>(iy)*N + iz ] += mass_ptr[i] * inv_vol;
                weight_idx[i] = {ix, iy, iz};

            }
        }
        else {
            throw std::runtime_error("boundary must be 'periodic' or 'isolated'");
        }

        // store for ordered weight output
    }

    // build Python weight list in original order
    py::list weights_list;
    for (ssize_t i = 0; i < M; ++i) {
        auto &idx3_arr = weight_idx[i];
        py::tuple idx3  = py::make_tuple(idx3_arr[0], idx3_arr[1], idx3_arr[2]);
        py::tuple entry = py::make_tuple(idx3, 1);
        py::list this_weights;
        this_weights.append(entry);
        weights_list.append(this_weights);
    }

    return std::make_pair(rho, weights_list);
}

// ---------------------- deposit_cic ----------------------
std::pair< py::array_t<double>, py::list > deposit_cic(
    py::array_t<double, py::array::c_style> positions,
    py::array_t<double, py::array::c_style> masses,
    int N,
    double box_size,
    const std::string &boundary)
{
    std::cout<<"Total threads: "<<omp_get_num_threads();
    
    auto buf_pos = positions.request();
    if (buf_pos.ndim != 2 || buf_pos.shape[1] != 3) {
        throw std::runtime_error("positions must be shape=(N,3) float64 C-contiguous ndarray");
    }
    auto buf_mass = masses.request();
    if (buf_mass.ndim != 1 || buf_mass.shape[0] != buf_pos.shape[0]) {
        throw std::runtime_error("masses must be shape=(N,) float64 C-contiguous ndarray");
    }

    double *pos_ptr  = static_cast<double*>(buf_pos.ptr);
    double *mass_ptr = static_cast<double*>(buf_mass.ptr);
    ssize_t M = buf_pos.shape[0];
    ssize_t cols = buf_pos.shape[1];

    py::array_t<double> rho({ (size_t)N, (size_t)N, (size_t)N });
    auto buf_rho = rho.request();
    double *rho_ptr = static_cast<double*>(buf_rho.ptr);
    ssize_t total_rho = (ssize_t)N * N * N;
    for (ssize_t i = 0; i < total_rho; ++i) {
        rho_ptr[i] = 0.0;
    }

    double dx      = box_size / static_cast<double>(N);
    double inv_vol = 1.0 / (dx * dx * dx);
    int NN = N * N;

    std::vector<std::vector<std::pair<std::array<int,3>, double>>> tmp(M);
    for (auto &v : tmp) {
        v.reserve(8);
    }

    #pragma omp parallel for 
    for (ssize_t i = 0; i < M; ++i) {
        double xp = pos_ptr[i * cols + 0] / dx;
        double yp = pos_ptr[i * cols + 1] / dx;
        double zp = pos_ptr[i * cols + 2] / dx;

        int i0 = static_cast<int>(std::floor(xp));
        int j0 = static_cast<int>(std::floor(yp));
        int k0 = static_cast<int>(std::floor(zp));

        double fx = xp - i0;
        double fy = yp - j0;
        double fz = zp - k0;

        double wx[2] = { 1.0 - fx, fx };
        double wy[2] = { 1.0 - fy, fy };
        double wz[2] = { 1.0 - fz, fz };

        for (int dx_ = 0; dx_ <= 1; ++dx_) {
            int ix = i0 + dx_;
            for (int dy_ = 0; dy_ <= 1; ++dy_) {
                int iy = j0 + dy_;
                for (int dz_ = 0; dz_ <= 1; ++dz_) {
                    int iz = k0 + dz_;
                    double w = wx[dx_] * wy[dy_] * wz[dz_];

                    if (boundary == "periodic") {
                        ix %= N; if (ix < 0) ix += N;
                        iy %= N; if (iy < 0) iy += N;
                        iz %= N; if (iz < 0) iz += N;
                        #pragma omp atomic
                        rho_ptr[ static_cast<ssize_t>(ix)*NN + static_cast<ssize_t>(iy)*N + iz ]
                            += mass_ptr[i] * inv_vol * w;
                        tmp[i].push_back({{ix,iy,iz}, w}); 

                    }
                    else if (boundary == "isolated") {
                        if (0 <= ix && ix < N && 0 <= iy && iy < N && 0 <= iz && iz < N) {
                            #pragma omp atomic
                            rho_ptr[ static_cast<ssize_t>(ix)*NN + static_cast<ssize_t>(iy)*N + iz ]
                                += mass_ptr[i] * inv_vol * w;
                            tmp[i].push_back({{ix,iy,iz}, w}); 

                        }
                    }
                    else {
                        throw std::runtime_error("boundary must be 'periodic' or 'isolated'");
                    }
                    
                }
            }
        }
    }
    py::list weights_list;
    for (ssize_t i = 0; i < M; ++i) {
        py::list this_weights;
        for (auto &p : tmp[i]) {
            auto &idx3 = p.first;
            double w  = p.second;
            py::tuple t = py::make_tuple(py::make_tuple(idx3[0], idx3[1], idx3[2]), w);
            this_weights.append(t);
        }
        weights_list.append(this_weights);
    }


    return std::make_pair(rho, weights_list);
}

// ---------------------- deposit_tsc ----------------------
std::pair< py::array_t<double>, py::list > deposit_tsc(
    py::array_t<double, py::array::c_style> positions,
    py::array_t<double, py::array::c_style> masses,
    int N,
    double box_size,
    const std::string &boundary)
{
    std::cout<<"Total threads: "<<omp_get_num_threads();

    auto buf_pos = positions.request();
    if (buf_pos.ndim != 2 || buf_pos.shape[1] != 3) {
        throw std::runtime_error("positions must be shape=(N,3) float64 C-contiguous ndarray");
    }
    auto buf_mass = masses.request();
    if (buf_mass.ndim != 1 || buf_mass.shape[0] != buf_pos.shape[0]) {
        throw std::runtime_error("masses must be shape=(N,) float64 C-contiguous ndarray");
    }

    double *pos_ptr  = static_cast<double*>(buf_pos.ptr);
    double *mass_ptr = static_cast<double*>(buf_mass.ptr);
    ssize_t M = buf_pos.shape[0];
    ssize_t cols = buf_pos.shape[1];

    py::array_t<double> rho({ (size_t)N, (size_t)N, (size_t)N });
    auto buf_rho = rho.request();
    double *rho_ptr = static_cast<double*>(buf_rho.ptr);
    ssize_t total_rho = (ssize_t)N * N * N;
    for (ssize_t i = 0; i < total_rho; ++i) {
        rho_ptr[i] = 0.0;
    }

    double dx      = box_size / static_cast<double>(N);
    double inv_vol = 1.0 / (dx * dx * dx);
    int NN = N * N;

    std::vector<std::vector<std::pair<std::array<int,3>, double>>> tmp(M);
    for (auto &v : tmp) {
        v.reserve(8);
    }

    #pragma omp parallel for
    for (ssize_t i = 0; i < M; ++i) {
        double xp = pos_ptr[i * cols + 0] / dx;
        double yp = pos_ptr[i * cols + 1] / dx;
        double zp = pos_ptr[i * cols + 2] / dx;

        int i0 = static_cast<int>(std::floor(xp+0.5));
        int j0 = static_cast<int>(std::floor(yp+0.5));
        int k0 = static_cast<int>(std::floor(zp+0.5));

        // For TSC kernel, we'll loop over i0-1, i0, i0+1 etc.
        for (int di = -1; di <= 1; ++di) {
            int ix = i0 + di;
            double rx = std::abs(xp - ix);
            double wx;
            if (rx < 0.5) {
                wx = 0.75 - rx*rx;
            }
            else if (rx < 1.5) {
                wx = 0.5 * (1.5 - rx) * (1.5 - rx);
            }
            else {
                wx = 0.0;
            }
            if (wx == 0.0) continue;

            for (int dj = -1; dj <= 1; ++dj) {
                int iy = j0 + dj;
                double ry = std::abs(yp - iy);
                double wy;
                if (ry < 0.5) {
                    wy = 0.75 - ry*ry;
                }
                else if (ry < 1.5) {
                    wy = 0.5 * (1.5 - ry) * (1.5 - ry);
                }
                else {
                    wy = 0.0;
                }
                if (wy == 0.0) continue;

                for (int dk = -1; dk <= 1; ++dk) {
                    int iz = k0 + dk;
                    double rz = std::abs(zp - iz);
                    double wz;
                    if (rz < 0.5) {
                        wz = 0.75 - rz*rz;
                    }
                    else if (rz < 1.5) {
                        wz = 0.5 * (1.5 - rz) * (1.5 - rz);
                    }
                    else {
                        wz = 0.0;
                    }
                    if (wz == 0.0) continue;

                    double w = wx * wy * wz;

                    if (boundary == "periodic") {
                        ix %= N; if (ix< 0) ix += N;
                        iy %= N; if (iy < 0) iy += N;
                        iz %= N; if (iz < 0) iz += N;
                        #pragma omp atomic
                        rho_ptr[ static_cast<ssize_t>(ix)*NN + static_cast<ssize_t>(iy)*N + iz ]
                            += mass_ptr[i] * inv_vol * w;
                        tmp[i].push_back({{ix,iy,iz}, w}); 

                    }
                    else if (boundary == "isolated") {
                        if (0 <= ix && ix < N && 0 <= iy && iy < N && 0 <= iz && iz < N) {
                            #pragma omp atomic
                            rho_ptr[ static_cast<ssize_t>(ix)*NN + static_cast<ssize_t>(iy)*N + iz ]
                                += mass_ptr[i] * inv_vol * w;
                            tmp[i].push_back({{ix,iy,iz}, w}); 

                        }
                    }
                    else {
                        throw std::runtime_error("boundary must be 'periodic' or 'isolated'");
                    }


                }
            }
        }

    }
    py::list weights_list;
    for (ssize_t i = 0; i < M; ++i) {
        py::list this_weights;
        for (auto &p : tmp[i]) {
            auto &idx3 = p.first;
            double w  = p.second;
            py::tuple t = py::make_tuple(py::make_tuple(idx3[0], idx3[1], idx3[2]), w);
            this_weights.append(t);
        }
        weights_list.append(this_weights);
    }


    return std::make_pair(rho, weights_list);
}



PYBIND11_MODULE(example_omp, m) {
    m.doc() = "Example: Use OpenMP to parallel fill and return a NumPy matrix";
    m.def("deposit_ngp",
          &deposit_ngp,
          "Return a deposit_ngp");
    m.def("deposit_cic",
          &deposit_cic,
          "Return a deposit_cic");
    m.def("deposit_tsc",
          &deposit_tsc,
          "Return a deposit_tsc");
}
