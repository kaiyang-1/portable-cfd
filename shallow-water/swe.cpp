//! Solves shallow water equations in 2D

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
    long nx, ny;
    float Lx, Ly, Tend;
    float dx, dt;

    static constexpr int halo = 4;

    parameters(int argc, char *argv[]) {
        if (argc != 5) {
            std::cerr << "ERROR: incorrect arguments" << std::endl;
            std::cerr << "  " << argv[0] << " <nx> <ny> <Lx> <Ly> <Tend>" << std::endl;
            std::terminate();
        }
        nx = std::stoll(argv[1]);
        ny = std::stoll(argv[2]);
        Lx = std::stof(argv[3]);
        Ly = std::stof(argv[4]);
        Tend = std::stof(argv[5]);
        dx = Lx / nx;
    }

    float Tout() { return 0.04; }
    long n() { return (ny + halo) * (nx + halo); }
};

// 2D grid of indicies
struct grid {
    long x_begin, x_end, y_begin, y_end;
};

// Initial condition
void initial_condition(float *h, float *u, float *v, parameters p) {
    grid g{.x_begin = p.halo / 2,
           .x_end = p.nx + p.halo / 2,
           .y_begin = p.halo / 2,
           .y_end = p.ny + p.halo / 2};

    auto xs = std::views::iota(g.x_begin, g.x_end);
    auto ys = std::views::iota(g.y_begin, g.y_end);
    auto ids = std::views::cartesian_product(xs, ys);

    std::for_each(std::execution::par, ids.begin(), ids.end(), [h, p](auto gid) {
        auto [i, j] = gid;

        auto idx = [=](auto i, auto j) { return j * (p.nx + p.halo) + i; };

        if (i * p.dx < 0.2f * p.Lx && j * p.dx < 0.2f * p.Ly)
            h[idx(i, j)] = 10.f;
        else
            h[idx(i, j)] = 1.f;
    });
    std::fill_n(std::execution::par, u, p.n(), 0.0);
    std::fill_n(std::execution::par, v, p.n(), 0.0);
}

// Boundary condition
void boundary_condition(float *h, float *u, float *v, parameters p) {
    {
        // left boundary
        grid g{.x_begin = 0, .x_end = p.halo / 2, .y_begin = 0, .y_end = p.ny + p.halo};
        auto xs = std::views::iota(g.x_begin, g.x_end);
        auto ys = std::views::iota(g.y_begin, g.y_end);
        auto ids = std::views::cartesian_product(xs, ys);

        std::for_each(std::execution::par, ids.begin(), ids.end(), [h, u, v, p](auto gid) {
            auto [i, j] = gid;

            auto idx = [=](auto i, auto j) { return j * (p.nx + p.halo) + i; };

            h[idx(i, j)] = h[idx(p.halo - 1 - i, j)];
            u[idx(i, j)] = -u[idx(p.halo - i, j)];
            if (i == p.halo / 2 - 1) {
                u[idx(i + 1, j)] = 0;
            }
            v[idx(i, j)] = v[idx(p.halo - 1 - i, j)];
        });
    }

    {
        // right boundary
        grid g{.x_begin = p.nx + p.halo / 2,
               .x_end = p.nx + p.halo,
               .y_begin = 0,
               .y_end = p.ny + p.halo};
        auto xs = std::views::iota(g.x_begin, g.x_end);
        auto ys = std::views::iota(g.y_begin, g.y_end);
        auto ids = std::views::cartesian_product(xs, ys);

        std::for_each(std::execution::par, ids.begin(), ids.end(), [h, u, v, p](auto gid) {
            auto [i, j] = gid;

            auto idx = [=](auto i, auto j) { return j * (p.nx + p.halo) + i; };
        });
    }
}

int main(int argc, char *argv[]) {
    // Parse CLI parameters
    parameters p(argc, argv);

    // Allocate memory
    std::vector<float> h(p.n()), u(p.n()), v(p.n());

    initial_condition(h.data(), u.data(), v.data(), p);
    boundary_condition(h.data(), u.data(), v.data(), p);

    return 0;
}