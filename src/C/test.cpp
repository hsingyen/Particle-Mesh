#include <iostream>
#include <random>
#include <vector>
#include <array>
#include <string>
#include "mass_deposition.hpp"  // 假設裡面有 deposit_cic, deposit_ngp, deposit_tsc 等函式宣告


// 用三重迴圈印出扁平化 rho
void print_rho_3d(const std::vector<double>& rho, int N, const std::string& name) {
    std::cout << name << " rho (3D):" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                int idx = i * N * N + j * N + k;
                std::cout << rho[idx] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << "--------------------------------" << std::endl;
}


int main() {
    const int N = 5;             
    const double box_size = 1.0;  
    const int N_particles = 2;    

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> pos_dist(0.0, box_size);   // 位置分布 [0, box_size]
    std::uniform_real_distribution<double> mass_dist(0.1, 1.0);        // 質量分布 [0.1, 1.0]
    std::uniform_real_distribution<double> vel_dist(-0.5, 0.5);        // 速度分布 [-0.5, 0.5]

    // --- 容器宣告 ---
    std::vector<std::array<double,3>> positions(N_particles);
    std::vector<double>                masses(N_particles);
    std::vector<std::array<double,3>> velocities(N_particles);
    
    // --- 初始化隨機資料 ---
    for (int i = 0; i < N_particles; ++i) {
        positions[i] = { pos_dist(gen), pos_dist(gen), pos_dist(gen) };
        std::cout<<positions[i][0]<<" "<<positions[i][1]<<" "<<positions[i][2]<<"\n";
        masses[i]    = 1.0;
        velocities[i]= {0.0, 0.0, 0.0};
    }

    auto ngp_res = deposit_ngp(positions, masses, N, box_size, "periodic");

    print_rho_3d(ngp_res.rho, N, "NGP");

    auto print_weights = [&](const GridDepositResult& res, const std::string& name){
        std::cout << name << " weights_list:\n";
        for (int p = 0; p < N_particles; ++p) {
            std::cout << " Particle " << p << ":";
            for (auto& wp : res.weights_list[p]) {
                const auto& idx = wp.first;
                double w = wp.second;
                std::cout << " (" << idx.x << "," << idx.y << "," << idx.z << "):"
                          << w;
            }
            std::cout << "\n";
        }
        std::cout << "================================\n";
    };

    print_weights(ngp_res, "NGP");

    return 0;

}
