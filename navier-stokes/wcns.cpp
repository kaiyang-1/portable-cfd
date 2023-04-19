//! Solves weakly compressible Navier-Stokes equations in 2D, see the README.

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

// DONE: add C++ standard library includes as necessary
#include <algorithm> // For std::fill_n
#include <execution> // For std::execution::par
#include <numeric>   // For std::transform_reduce
#include <ranges>    // For std::views

// Problem parameters
struct parameters {
    double dx, dt;
    long nx, ny, ni;

    double nu, Re, Ma;

    // Reads command line arguments to initialize problem size
    parameters(int argc, char *argv[]) {
        if (argc != 5) {
            std::cerr << "ERROR: incorrect arguments" << std::endl;
            std::cerr << "  " << argv[0] << " <nx> <ny> <step> <Re>" << std::endl;
            std::terminate();
        }
        nx = std::stoll(argv[1]);
        ny = std::stoll(argv[2]);
        ni = std::stoll(argv[3]);
        Re = std::stof(argv[4]);

        dx = 1. / nx;
        nu = 1. / Re;

        Ma = 0.05;

        int dim = 2;

        float dt_v = Ma * dx;
        float dt_nu = dx * dx / (2. * dim * nu);

        dt = dt_v < dt_nu ? dt_v : dt_nu;
    }

    long nit() { return ni; }
    long n() { return (ny + 2) * (nx + 2); /* 2 halo layers */ }
};

double stencil(double *u_new, double *u_old, long x, long y, parameters p);

// 2D grid of indicies
struct grid {
    long x_begin, x_end, y_begin, y_end;
};

double apply_stencil(double *u_new, double *u_old, grid g, parameters p) {
    auto xs = std::views::iota(g.x_begin, g.x_end);
    auto ys = std::views::iota(g.y_begin, g.y_end);
    auto ids = std::views::cartesian_product(xs, ys);
    return std::transform_reduce(std::execution::par, ids.begin(), ids.end(), 0., std::plus{},
                                 [u_new, u_old, p](auto idx) {
                                     auto [x, y] = idx;
                                     return stencil(u_new, u_old, x, y, p);
                                 });
}

// Initial condition
void initial_condition(double *u_new, double *u_old, long n) {
    std::fill_n(std::execution::par, u_old, n, 0.0);
    std::fill_n(std::execution::par, u_new, n, 0.0);
}

// These evolve the solution of different parts of the local domain.
double inner(double *u_new, double *u_old, parameters p);

int main(int argc, char *argv[]) {
    // Parse CLI parameters
    parameters p(argc, argv);

    // Allocate memory
    std::vector<double> u_new(p.n()), u_old(p.n());

    // Initial condition
    initial_condition(u_new.data(), u_old.data(), p.n());

    // Time loop
    using clk_t = std::chrono::steady_clock;
    auto start = clk_t::now();

    for (long it = 0; it < p.nit(); ++it) {
        // Evolve the solution:
        double energy = inner(u_new.data(), u_old.data(), p);

        std::swap(u_new, u_old);
    }

    auto time = std::chrono::duration<double>(clk_t::now() - start).count();
    auto grid_size = static_cast<double>(p.nx * p.ny * sizeof(double) * 2) * 1e-9; // GB
    auto memory_bw = grid_size * static_cast<double>(p.nit()) / time;              // GB/s
    std::cerr << "Domain " << p.nx << "x" << p.ny << " (" << grid_size << " GB): " << memory_bw
              << " GB/s" << std::endl;

    return 0;
}

// Finite-difference stencil
double stencil(double *u_new, double *u_old, long x, long y, parameters p) {
    auto idx = [=](auto x, auto y) {
        // Index into the memory using row-major order:
        assert(x >= 0 && x < 2 * p.nx);
        assert(y >= 0 && y < p.ny);
        return x * p.ny + y;
    };
    // Apply boundary conditions:
    if (y == 1) {
        u_old[idx(x, y - 1)] = 0;
    }
    if (y == (p.ny - 2)) {
        u_old[idx(x, y + 1)] = 0;
    }

    u_new[idx(x, y)] = (1. - 4. * p.gamma()) * u_old[idx(x, y)] +
                       p.gamma() * (u_old[idx(x + 1, y)] + u_old[idx(x - 1, y)] +
                                    u_old[idx(x, y + 1)] + u_old[idx(x, y - 1)]);

    return u_new[idx(x, y)] * p.dx * p.dx;
}

double inner(double *u_new, double *u_old, parameters p) {
    grid g{.x_begin = 2, .x_end = p.nx, .y_begin = 1, .y_end = p.ny - 1};
    return apply_stencil(u_new, u_old, g, p);
}