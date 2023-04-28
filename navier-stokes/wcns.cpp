//! Solves weakly compressible equations in 2D

#include "../include/cartesian_product.hpp"
#include "../include/scheme.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
// DONE: add C++ standard library includes as necessary
#include <algorithm>
#include <cmath>
#include <execution>
#include <limits>
#include <numeric>

using uint = unsigned int;

// Problem parameters
struct parameters {
    uint nx, ny, ni;
    float Re, Cs;
    float dx, dt;

    static constexpr float Ma = 0.05;
    static constexpr uint halo = 4;

    parameters(int argc, char *argv[]) {
        if (argc != 5) {
            std::cerr << "ERROR: incorrect arguments" << std::endl;
            std::cerr << "  " << argv[0] << " <nx> <ny> <ni> <Re>" << std::endl;
            std::terminate();
        }
        nx = std::stoull(argv[1]);
        ny = std::stoull(argv[2]);
        ni = std::stoull(argv[3]);
        Re = std::stof(argv[4]);

        dx = 1.0 / nx;
    }

    uint nit() { return ni; }
    uint nout() { return 1000; }
    uint n() { return (ny + halo) * (nx + halo); }
};

// 2D grid of indicies
struct grid {
    uint x_begin, x_end, y_begin, y_end;
};

// Boundary condition
void boundary_pressure(float *p, parameters par) {
    {
        // x boundary
        grid g{.x_begin = 0, .x_end = par.halo, .y_begin = 0, .y_end = par.ny + par.halo};
        auto xs = std::views::iota(g.x_begin, g.x_end);
        auto ys = std::views::iota(g.y_begin, g.y_end);
        auto ids = std::views::cartesian_product(xs, ys);

        std::for_each(std::execution::par, ids.begin(), ids.end(), [p, par](auto gid) {
            auto idx = [=](auto i, auto j) { return j * (par.nx + par.halo) + i; };
            auto [i, j] = gid;

            if (i < par.halo / 2) {
                // left
                p[idx(i, j)] = p[idx(par.halo - 1 - i, j)];
            } else {
                // right
                i += par.nx;
                p[idx(i, j)] = p[idx(2 * par.nx + par.halo - 1 - i, j)];
            }
        });
    }

    {
        // y boundary
        grid g{.x_begin = 0, .x_end = par.nx + par.halo, .y_begin = 0, .y_end = par.halo};
        auto xs = std::views::iota(g.x_begin, g.x_end);
        auto ys = std::views::iota(g.y_begin, g.y_end);
        auto ids = std::views::cartesian_product(xs, ys);

        std::for_each(std::execution::par, ids.begin(), ids.end(), [p, par](auto gid) {
            auto idx = [=](auto i, auto j) { return j * (par.nx + par.halo) + i; };
            auto [i, j] = gid;

            if (j < par.halo / 2) {
                // bottom
                p[idx(i, j)] = p[idx(i, par.halo - 1 - j)];
            } else {
                // top
                j += par.ny;
                p[idx(i, j)] = p[idx(i, 2 * par.ny + par.halo - 1 - j)];
            }
        });
    }
}

// Boundary condition
void boundary_velocity(float *u, float *v, parameters par) {
    {
        // x boundary
        grid g{.x_begin = 0, .x_end = par.halo, .y_begin = 0, .y_end = par.ny + par.halo};
        auto xs = std::views::iota(g.x_begin, g.x_end);
        auto ys = std::views::iota(g.y_begin, g.y_end);
        auto ids = std::views::cartesian_product(xs, ys);

        std::for_each(std::execution::par, ids.begin(), ids.end(), [u, v, par](auto gid) {
            auto idx = [=](auto i, auto j) { return j * (par.nx + par.halo) + i; };
            auto [i, j] = gid;

            if (i < par.halo / 2) {
                // left
                u[idx(i, j)] = -u[idx(par.halo - i, j)];
                if (i == par.halo / 2 - 1) {
                    u[idx(i + 1, j)] = 0;
                }
                v[idx(i, j)] = -v[idx(par.halo - 1 - i, j)];
            } else {
                // right
                i += par.nx;
                if (i == par.nx + par.halo / 2) {
                    u[idx(i, j)] = 0;
                } else {
                    u[idx(i, j)] = -u[idx(2 * par.nx + par.halo - i, j)];
                }
                v[idx(i, j)] = -v[idx(2 * par.nx + par.halo - 1 - i, j)];
            }
        });
    }

    {
        // y boundary
        grid g{.x_begin = 0, .x_end = par.nx + par.halo, .y_begin = 0, .y_end = par.halo};
        auto xs = std::views::iota(g.x_begin, g.x_end);
        auto ys = std::views::iota(g.y_begin, g.y_end);
        auto ids = std::views::cartesian_product(xs, ys);

        std::for_each(std::execution::par, ids.begin(), ids.end(), [u, v, par](auto gid) {
            auto idx = [=](auto i, auto j) { return j * (par.nx + par.halo) + i; };
            auto [i, j] = gid;

            if (j < par.halo / 2) {
                // bottom
                u[idx(i, j)] = -u[idx(i, par.halo - 1 - j)];
                v[idx(i, j)] = -v[idx(i, par.halo - j)];
                if (j == par.halo / 2 - 1) {
                    v[idx(i, j + 1)] = 0;
                }
            } else {
                // top
                j += par.ny;
                u[idx(i, j)] = 1.0; // Lid-driven cavity flow
                if (j == par.ny + par.halo / 2) {
                    v[idx(i, j)] = 0;
                } else {
                    v[idx(i, j)] = -v[idx(i, 2 * par.ny + par.halo - j)];
                }
            }
        });
    }
}

// Initial condition
void initial_condition(float *p, float *u, float *v, parameters par) {
    std::fill_n(std::execution::par, p, par.n(), 0.0);
    std::fill_n(std::execution::par, u, par.n(), 0.0);
    std::fill_n(std::execution::par, v, par.n(), 0.0);
}

void pressure_stencil(float *p_old, float *p_new, float *div_u, const float rho, parameters par) {
    grid g{.x_begin = par.halo / 2,
           .x_end = par.nx + par.halo / 2,
           .y_begin = par.halo / 2,
           .y_end = par.ny + par.halo / 2};

    auto xs = std::views::iota(g.x_begin, g.x_end);
    auto ys = std::views::iota(g.y_begin, g.y_end);
    auto ids = std::views::cartesian_product(xs, ys);

    std::for_each(std::execution::par, ids.begin(), ids.end(),
                  [p_old, p_new, div_u, par, rho](auto gid) {
                      auto idx = [=](auto i, auto j) { return j * (par.nx + par.halo) + i; };
                      auto [i, j] = gid;

                      p_new[idx(i, j)] =
                          p_old[idx(i, j)] -
                          par.dt * rho * par.Cs * par.Cs *
                              (div_u[idx(i, j)] -
                               (scheme::central_diff_2nd(p_old[idx(i - 1, j)], p_old[idx(i, j)],
                                                         p_old[idx(i + 1, j)], par.dx) +
                                scheme::central_diff_2nd(p_old[idx(i, j - 1)], p_old[idx(i, j)],
                                                         p_old[idx(i, j + 1)], par.dx)) /
                                   rho * par.dt);
                  });
}

void pressure_evolution(float *p_old, float *p_new, float *u, float *v, parameters par) {
    std::vector<float> div_u(par.n());
    constexpr float rho = 1;

    grid g{.x_begin = par.halo / 2,
           .x_end = par.nx + par.halo / 2,
           .y_begin = par.halo / 2,
           .y_end = par.ny + par.halo / 2};

    auto xs = std::views::iota(g.x_begin, g.x_end);
    auto ys = std::views::iota(g.y_begin, g.y_end);
    auto ids = std::views::cartesian_product(xs, ys);

    std::for_each(
        std::execution::par, ids.begin(), ids.end(), [div_u = div_u.data(), u, v, par](auto gid) {
            auto idx = [=](auto i, auto j) { return j * (par.nx + par.halo) + i; };
            auto [i, j] = gid;

            div_u[idx(i, j)] =
                (u[idx(i + 1, j)] - u[idx(i, j)] + v[idx(i, j + 1)] - v[idx(i, j)]) / par.dx;
        });

    for (int iter = 0; iter < 20; iter++) {
        pressure_stencil(p_old, p_new, div_u.data(), rho, par);
        boundary_pressure(p_new, par);
        std::swap(p_new, p_old);
    }

    std::for_each(std::execution::par, ids.begin(), ids.end(), [u, v, p_old, par, rho](auto gid) {
        auto idx = [=](auto i, auto j) { return j * (par.nx + par.halo) + i; };
        auto [i, j] = gid;

        u[idx(i, j)] -= (p_old[idx(i, j)] - p_old[idx(i - 1, j)]) / par.dx * par.dt / rho;
        v[idx(i, j)] -= (p_old[idx(i, j)] - p_old[idx(i, j - 1)]) / par.dx * par.dt / rho;
    });
}

float momentum_stencil(float *u_new, float *v_new, float *u_old, float *v_old, float *u_n,
                       float *v_n, parameters par, float c0, float c1) {
    grid g{.x_begin = par.halo / 2,
           .x_end = par.nx + par.halo / 2,
           .y_begin = par.halo / 2,
           .y_end = par.ny + par.halo / 2};

    auto xs = std::views::iota(g.x_begin, g.x_end);
    auto ys = std::views::iota(g.y_begin, g.y_end);
    auto ids = std::views::cartesian_product(xs, ys);

    float max_vel = std::transform_reduce(
        std::execution::par, ids.begin(), ids.end(), std::numeric_limits<float>::min(),
        [](auto a, auto b) { return std::max(a, b); },
        [u_new, v_new, u_old, v_old, u_n, v_n, par, c0, c1](auto gid) {
            auto idx = [=](auto i, auto j) { return j * (par.nx + par.halo) + i; };
            auto [i, j] = gid;

            float u_adv, v_adv, adv_term, visc_term;
            int upwind;

            // update u
            u_adv = u_old[idx(i, j)];
            v_adv = 0.25f * (v_old[idx(i - 1, j)] + v_old[idx(i, j)] + v_old[idx(i - 1, j + 1)] +
                             v_old[idx(i, j + 1)]);

            adv_term = 0;
            upwind = u_adv < 0 ? 1 : -1;
            adv_term += u_adv * scheme::HJ_WENO3(u_old[idx(i - upwind, j)], u_old[idx(i, j)],
                                                 u_old[idx(i + upwind, j)],
                                                 u_old[idx(i + 2 * upwind, j)], u_adv, par.dx);
            upwind = v_adv < 0 ? 1 : -1;
            adv_term += v_adv * scheme::HJ_WENO3(u_old[idx(i, j - upwind)], u_old[idx(i, j)],
                                                 u_old[idx(i, j + upwind)],
                                                 u_old[idx(i, j + 2 * upwind)], v_adv, par.dx);
            visc_term = (scheme::central_diff_2nd(u_old[idx(i - 1, j)], u_old[idx(i, j)],
                                                  u_old[idx(i + 1, j)], par.dx) +
                         scheme::central_diff_2nd(u_old[idx(i, j - 1)], u_old[idx(i, j)],
                                                  u_old[idx(i, j + 1)], par.dx)) /
                        par.Re;

            u_new[idx(i, j)] =
                c0 * u_n[idx(i, j)] + c1 * (u_old[idx(i, j)] + (visc_term - adv_term) * par.dt);

            // update v
            u_adv = 0.25f * (u_old[idx(i, j - 1)] + u_old[idx(i, j)] + u_old[idx(i + 1, j - 1)] +
                             u_old[idx(i + 1, j)]);
            v_adv = v_old[idx(i, j)];

            adv_term = 0;
            upwind = u_adv < 0 ? 1 : -1;
            adv_term += u_adv * scheme::HJ_WENO3(v_old[idx(i - upwind, j)], v_old[idx(i, j)],
                                                 v_old[idx(i + upwind, j)],
                                                 v_old[idx(i + 2 * upwind, j)], u_adv, par.dx);
            upwind = v_adv < 0 ? 1 : -1;
            adv_term += v_adv * scheme::HJ_WENO3(v_old[idx(i, j - upwind)], v_old[idx(i, j)],
                                                 v_old[idx(i, j + upwind)],
                                                 v_old[idx(i, j + 2 * upwind)], v_adv, par.dx);
            visc_term = (scheme::central_diff_2nd(v_old[idx(i - 1, j)], v_old[idx(i, j)],
                                                  v_old[idx(i + 1, j)], par.dx) +
                         scheme::central_diff_2nd(v_old[idx(i, j - 1)], v_old[idx(i, j)],
                                                  v_old[idx(i, j + 1)], par.dx)) /
                        par.Re;

            v_new[idx(i, j)] =
                c0 * v_n[idx(i, j)] + c1 * (v_old[idx(i, j)] + (visc_term - adv_term) * par.dt);

            return std::abs(u_new[idx(i, j)]) + std::abs(v_new[idx(i, j)]);
        });

    return max_vel;
}

float solve_momentum(float *u_old, float *v_old, float *u_new, float *v_new, float *u_rk,
                     float *v_rk, parameters par) {
    float max_vel;

    // 3rd-order 3-stage TVD Runge-Kutta method
    // 1st stage u_old, v_old --> u_new, v_new
    max_vel = momentum_stencil(u_new, v_new, u_old, v_old, u_old, v_old, par, 0.f, 1.f);
    boundary_velocity(u_new, v_new, par);

    // 2nd stage u_new, v_new --> u_rk, v_rk
    max_vel = momentum_stencil(u_rk, v_rk, u_new, v_new, u_old, v_old, par, 3.f / 4.f, 1.f / 4.f);
    boundary_velocity(u_rk, v_rk, par);

    // 3rd stage u_rk, v_rk --> u_new, v_new
    max_vel = momentum_stencil(u_new, v_new, u_rk, v_rk, u_old, v_old, par, 1.f / 3.f, 2.f / 3.f);
    boundary_velocity(u_new, v_new, par);

    std::swap(u_new, u_old);
    std::swap(v_new, v_old);

    return max_vel;
}

void write_to_csv(float *u, float *v, parameters par, std::string filename);

int main(int argc, char *argv[]) {
    // Parse CLI parameters
    parameters par(argc, argv);

    const float CFL = 0.8;

    // Allocate memory
    std::vector<float> p_old(par.n()), p_new(par.n());
    std::vector<float> u_old(par.n()), u_new(par.n()), u_rk(par.n());
    std::vector<float> v_old(par.n()), v_new(par.n()), v_rk(par.n());

    initial_condition(p_old.data(), u_old.data(), v_old.data(), par);
    par.dt = CFL * par.Ma * par.dx;

    // main loop
    for (uint it = 0; it < par.nit(); ++it) {
        float v_max = solve_momentum(u_old.data(), v_old.data(), u_new.data(), v_new.data(),
                                     u_rk.data(), v_rk.data(), par);
        pressure_evolution(p_old.data(), p_new.data(), u_old.data(), v_old.data(), par);

        if (it % par.nout() == 0) {
            std::cout << "Steps = " << it << ", max vel = " << v_max << std::endl;
        }

        par.dt = CFL * par.Ma * par.dx / v_max;
    }

    write_to_csv(u_old.data(), v_old.data(), par, "cavity_flow.csv");

    return 0;
}

void write_to_csv(float *u, float *v, parameters par, std::string filename) {
    auto idx = [=](auto x, auto y) { return y * (par.nx + par.halo) + x; };

    std::ofstream file(filename);

    for (int j = par.halo / 2; j < par.ny + par.halo / 2; j++) {
        for (int i = par.halo / 2; i < par.nx + par.halo / 2; i++) {
            if (i != par.halo / 2) {
                file << ",";
            }
            float u_c = 0.5 * (u[idx(i, j)] + u[idx(i + 1, j)]);
            float v_c = 0.5 * (v[idx(i, j)] + v[idx(i, j + 1)]);
            file << std::sqrt(u_c * u_c + v_c * v_c);
        }
        file << std::endl;
    }

    file.close();
}