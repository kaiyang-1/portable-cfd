//! Solves heat equation in 2D, see the README.

#include "../include/cartesian_product.hpp"
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
// DONE: add C++ standard library includes as necessary
#include <algorithm> // For std::fill_n
// #include <ranges>    // For std::views
#include <execution> // For std::execution::par
#include <numeric>   // For std::transform_reduce

// Problem parameters
struct parameters {
    float dx, dt;
    long nx, ny, ni;

    static constexpr int halo = 2;
    static constexpr float alpha() { return 1.0; } // Thermal diffusivity

    parameters() = default;
    ~parameters() = default;
    parameters(int argc, char *argv[]);

    constexpr parameters(const parameters &) = default;            // copy ctor
    constexpr parameters &operator=(const parameters &) = default; // copy-assign ctor
    constexpr parameters(parameters &&) = default;                 // move ctor
    constexpr parameters &operator=(parameters &&) = default;      // move-assign ctor

    long nit() { return ni; }
    long nout() { return 1000; }
    float gamma() { return alpha() * dt / (dx * dx); }
    long n() { return (ny + halo) * (nx + halo); }
};

float stencil(float *u_new, float *u_old, long x, long y, parameters par);

void write_to_csv(float *u, parameters par, std::string filename);

// 2D grid of indicies
struct grid {
    long x_begin, x_end, y_begin, y_end;
};

float apply_stencil(float *u_new, float *u_old, grid g, parameters par) {
    auto xs = std::views::iota(g.x_begin, g.x_end);
    auto ys = std::views::iota(g.y_begin, g.y_end);
    auto ids = std::views::cartesian_product(xs, ys);
    return std::transform_reduce(std::execution::par, ids.begin(), ids.end(), 0., std::plus{},
                                 [u_new, u_old, par](auto idx) {
                                     auto [x, y] = idx;
                                     return stencil(u_new, u_old, x, y, par);
                                 });
}

// Initial condition
void initial_condition(float *u_new, float *u_old, long n) {
    std::fill_n(std::execution::par, u_old, n, 0.0);
    std::fill_n(std::execution::par, u_new, n, 0.0);
}

// These evolve the solution of different parts of the local domain.
float inner(float *u_new, float *u_old, parameters par);

int main(int argc, char *argv[]) {
    // Parse CLI parameters
    parameters par(argc, argv);

    // Allocate memory
    std::vector<float> u_new(par.n()), u_old(par.n());

    // Initial condition
    initial_condition(u_new.data(), u_old.data(), par.n());

    // Time loop
    using clk_t = std::chrono::steady_clock;
    auto start = clk_t::now();

    for (long it = 0; it < par.nit(); ++it) {
        // Evolve the solution:
        float energy = inner(u_new.data(), u_old.data(), par);

        if (it % par.nout() == 0) {
            std::cerr << "E(t=" << it * par.dt << ") = " << energy << std::endl;
        }
        std::swap(u_new, u_old);
    }

    auto time = std::chrono::duration<float>(clk_t::now() - start).count();
    auto grid_size = static_cast<float>(par.nx * par.ny) * 1e-6;   // cells
    auto mcups = grid_size * static_cast<float>(par.nit()) / time; // cell update per second
    std::cerr << "Local domain " << par.nx << "x" << par.ny << " (" << grid_size
              << " million cells): " << mcups << " MCUPS" << std::endl;

    write_to_csv(u_old.data(), par, "heat_data.csv");

    return 0;
}

// Reads command line arguments to initialize problem size
parameters::parameters(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "ERROR: incorrect arguments" << std::endl;
        std::cerr << "  " << argv[0] << " <nx> <ny> <ni>" << std::endl;
        std::terminate();
    }
    nx = std::stoll(argv[1]);
    ny = std::stoll(argv[2]);
    ni = std::stoll(argv[3]);
    dx = 1.0 / nx;
    dt = dx * dx / (5. * alpha());
}

// Finite-difference stencil
float stencil(float *u_new, float *u_old, long x, long y, parameters par) {
    auto idx = [=](auto x, auto y) {
        // Index into the memory using row-major order:
        return y * (par.nx + par.halo) + x;
    };
    // Apply boundary conditions:
    if (y == par.halo / 2) {
        u_old[idx(x, y - 1)] = 1;
    }
    if (y == (par.ny - 1 + par.halo / 2)) {
        u_old[idx(x, y + 1)] = 0;
    }
    if (x == par.halo / 2) {
        u_old[idx(x - 1, y)] = 0;
    }
    if (x == par.nx - 1 + par.halo / 2) {
        u_old[idx(x + 1, y)] = 0;
    }

    u_new[idx(x, y)] = (1. - 4. * par.gamma()) * u_old[idx(x, y)] +
                       par.gamma() * (u_old[idx(x + 1, y)] + u_old[idx(x - 1, y)] +
                                      u_old[idx(x, y + 1)] + u_old[idx(x, y - 1)]);

    return u_new[idx(x, y)] * par.dx * par.dx;
}

// Evolve the solution of the interior part of the domain
// which does not depend on data from neighboring ranks
float inner(float *u_new, float *u_old, parameters par) {
    grid g{.x_begin = par.halo / 2,
           .x_end = par.nx + par.halo / 2,
           .y_begin = par.halo / 2,
           .y_end = par.ny + par.halo / 2};
    return apply_stencil(u_new, u_old, g, par);
}

void write_to_csv(float *u, parameters par, std::string filename) {
    auto idx = [=](auto x, auto y) {
        // Index into the memory using row-major order:
        return y * (par.nx + par.halo) + x;
    };

    std::ofstream file(filename);

    for (int j = 0; j < par.ny; j++) {
        for (int i = 0; i < par.nx; i++) {
            if (i != 0) {
                file << ",";
            }
            file << u[idx(i + par.halo / 2, j + par.halo / 2)];
        }
        file << std::endl;
    }

    file.close();
}