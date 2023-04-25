//! Solves shallow water equations in 2D

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

// Problem parameters
struct parameters {
    uint nx, ny;
    float Lx, Ly, Tend;
    float dx, dt;

    static constexpr uint halo = 4;
    static constexpr float gravity = 9.8;

    parameters(int argc, char *argv[]) {
        if (argc != 6) {
            std::cerr << "ERROR: incorrect arguments" << std::endl;
            std::cerr << "  " << argv[0] << " <nx> <ny> <Lx> <Ly> <Tend>" << std::endl;
            std::terminate();
        }
        nx = std::stoull(argv[1]);
        ny = std::stoull(argv[2]);
        Lx = std::stof(argv[3]);
        Ly = std::stof(argv[4]);
        Tend = std::stof(argv[5]);
        dx = Lx / nx;
    }

    float Tout() { return 0.04; }
    uint n() { return (ny + halo) * (nx + halo); }
};

// 2D grid of indicies
struct grid {
    uint x_begin, x_end, y_begin, y_end;
};

// Boundary condition
void boundary_height(float *h, parameters p) {
    {
        // x boundary
        grid g{.x_begin = 0, .x_end = p.halo, .y_begin = 0, .y_end = p.ny + p.halo};
        auto xs = std::views::iota(g.x_begin, g.x_end);
        auto ys = std::views::iota(g.y_begin, g.y_end);
        auto ids = std::views::cartesian_product(xs, ys);

        std::for_each(std::execution::par, ids.begin(), ids.end(), [h, p](auto gid) {
            auto idx = [=](auto i, auto j) { return j * (p.nx + p.halo) + i; };
            auto [i, j] = gid;

            if (i < p.halo / 2) {
                // left
                h[idx(i, j)] = h[idx(p.halo - 1 - i, j)];
            } else {
                // right
                i += p.nx;
                h[idx(i, j)] = h[idx(2 * p.nx + p.halo - 1 - i, j)];
            }
        });
    }

    {
        // y boundary
        grid g{.x_begin = 0, .x_end = p.nx + p.halo, .y_begin = 0, .y_end = p.halo};
        auto xs = std::views::iota(g.x_begin, g.x_end);
        auto ys = std::views::iota(g.y_begin, g.y_end);
        auto ids = std::views::cartesian_product(xs, ys);

        std::for_each(std::execution::par, ids.begin(), ids.end(), [h, p](auto gid) {
            auto idx = [=](auto i, auto j) { return j * (p.nx + p.halo) + i; };
            auto [i, j] = gid;

            if (j < p.halo / 2) {
                // bottom
                h[idx(i, j)] = h[idx(i, p.halo - 1 - j)];
            } else {
                // top
                j += p.ny;
                h[idx(i, j)] = h[idx(i, 2 * p.ny + p.halo - 1 - j)];
            }
        });
    }
}

// Boundary condition
void boundary_velocity(float *u, float *v, parameters p) {
    {
        // x boundary
        grid g{.x_begin = 0, .x_end = p.halo, .y_begin = 0, .y_end = p.ny + p.halo};
        auto xs = std::views::iota(g.x_begin, g.x_end);
        auto ys = std::views::iota(g.y_begin, g.y_end);
        auto ids = std::views::cartesian_product(xs, ys);

        std::for_each(std::execution::par, ids.begin(), ids.end(), [u, v, p](auto gid) {
            auto idx = [=](auto i, auto j) { return j * (p.nx + p.halo) + i; };
            auto [i, j] = gid;

            if (i < p.halo / 2) {
                // left
                u[idx(i, j)] = -u[idx(p.halo - i, j)];
                if (i == p.halo / 2 - 1) {
                    u[idx(i + 1, j)] = 0;
                }
                v[idx(i, j)] = v[idx(p.halo - 1 - i, j)];
            } else {
                // right
                i += p.nx;
                if (i == p.nx + p.halo / 2) {
                    u[idx(i, j)] = 0;
                } else {
                    u[idx(i, j)] = -u[idx(2 * p.nx + p.halo - i, j)];
                }
                v[idx(i, j)] = v[idx(2 * p.nx + p.halo - 1 - i, j)];
            }
        });
    }

    {
        // y boundary
        grid g{.x_begin = 0, .x_end = p.nx + p.halo, .y_begin = 0, .y_end = p.halo};
        auto xs = std::views::iota(g.x_begin, g.x_end);
        auto ys = std::views::iota(g.y_begin, g.y_end);
        auto ids = std::views::cartesian_product(xs, ys);

        std::for_each(std::execution::par, ids.begin(), ids.end(), [u, v, p](auto gid) {
            auto idx = [=](auto i, auto j) { return j * (p.nx + p.halo) + i; };
            auto [i, j] = gid;

            if (j < p.halo / 2) {
                // bottom
                u[idx(i, j)] = u[idx(i, p.halo - 1 - j)];
                v[idx(i, j)] = -v[idx(i, p.halo - j)];
                if (j == p.halo / 2 - 1) {
                    v[idx(i, j + 1)] = 0;
                }
            } else {
                // top
                j += p.ny;
                u[idx(i, j)] = u[idx(i, 2 * p.ny + p.halo - 1 - j)];
                if (j == p.ny + p.halo / 2) {
                    v[idx(i, j)] = 0;
                } else {
                    v[idx(i, j)] = -v[idx(i, 2 * p.ny + p.halo - j)];
                }
            }
        });
    }
}

// Initial condition
void initial_condition(float *h, float *u, float *v, parameters p, float init_h) {
    grid g{.x_begin = p.halo / 2,
           .x_end = p.nx + p.halo / 2,
           .y_begin = p.halo / 2,
           .y_end = p.ny + p.halo / 2};

    auto xs = std::views::iota(g.x_begin, g.x_end);
    auto ys = std::views::iota(g.y_begin, g.y_end);
    auto ids = std::views::cartesian_product(xs, ys);

    std::for_each(std::execution::par, ids.begin(), ids.end(), [h, p, init_h](auto gid) {
        auto idx = [=](auto i, auto j) { return j * (p.nx + p.halo) + i; };
        auto [i, j] = gid;

        float wx = (i - p.halo / 2 + 0.5f) * p.dx;
        float wy = (j - p.halo / 2 + 0.5f) * p.dx;
        /*
        float radius = 0.1 * p.Lx;
        float dis = std::sqrt((wx - 0.5f * p.Lx) * (wx - 0.5f * p.Lx) +
                              (wy - 0.5f * p.Ly) * (wy - 0.5f * p.Ly));
        if (dis <= radius) {
            h[idx(i, j)] = init_h * std::cos(dis / radius * 0.5f * M_PI);
        */
        if (wx < 0.2 * p.Lx && wy < 0.2 * p.Ly) {
            h[idx(i, j)] = init_h;
        } else
            h[idx(i, j)] = 0;
    });

    boundary_height(h, p);

    std::fill_n(std::execution::par, u, p.n(), 0.0);
    std::fill_n(std::execution::par, v, p.n(), 0.0);
}

void height_flux(float *flx, float *fly, float *h, float *u, float *v, parameters p) {
    grid g{.x_begin = p.halo / 2,
           .x_end = p.nx + p.halo / 2 + 1,
           .y_begin = p.halo / 2,
           .y_end = p.ny + p.halo / 2 + 1};

    auto xs = std::views::iota(g.x_begin, g.x_end);
    auto ys = std::views::iota(g.y_begin, g.y_end);
    auto ids = std::views::cartesian_product(xs, ys);

    std::for_each(std::execution::par, ids.begin(), ids.end(), [flx, fly, h, u, v, p](auto gid) {
        auto idx = [=](auto i, auto j) { return j * (p.nx + p.halo) + i; };
        auto [i, j] = gid;

        float u_adv = u[idx(i, j)];
        float v_adv = v[idx(i, j)];

        if (u_adv < 0) {
            flx[idx(i, j)] =
                u_adv * scheme::TVD_MUSCL3(h[idx(i - 1, j)], h[idx(i, j)], h[idx(i + 1, j)]);
        } else {
            flx[idx(i, j)] =
                u_adv * scheme::TVD_MUSCL3(h[idx(i, j)], h[idx(i - 1, j)], h[idx(i - 2, j)]);
        }

        if (v_adv < 0) {
            fly[idx(i, j)] =
                v_adv * scheme::TVD_MUSCL3(h[idx(i, j - 1)], h[idx(i, j)], h[idx(i, j + 1)]);
        } else {
            fly[idx(i, j)] =
                v_adv * scheme::TVD_MUSCL3(h[idx(i, j)], h[idx(i, j - 1)], h[idx(i, j - 2)]);
        }
    });
}

float height_integral(float *h_new, float *h_old, float *h_n, float *flx, float *fly, parameters p,
                      float c0, float c1) {
    grid g{.x_begin = p.halo / 2,
           .x_end = p.nx + p.halo / 2,
           .y_begin = p.halo / 2,
           .y_end = p.ny + p.halo / 2};

    auto xs = std::views::iota(g.x_begin, g.x_end);
    auto ys = std::views::iota(g.y_begin, g.y_end);
    auto ids = std::views::cartesian_product(xs, ys);

    float max_height = std::transform_reduce(
        std::execution::par, ids.begin(), ids.end(), std::numeric_limits<float>::min(),
        [](auto a, auto b) { return std::max(a, b); },
        [h_new, h_old, h_n, flx, fly, p, c0, c1](auto gid) {
            auto idx = [=](auto i, auto j) { return j * (p.nx + p.halo) + i; };
            auto [i, j] = gid;

            h_new[idx(i, j)] = c0 * h_n[idx(i, j)] +
                               c1 * (h_old[idx(i, j)] + (flx[idx(i, j)] - flx[idx(i + 1, j)] +
                                                         fly[idx(i, j)] - fly[idx(i, j + 1)]) /
                                                            p.dx * p.dt);

            return h_new[idx(i, j)];
        });

    return max_height;
}

float solve_height(float *h_old, float *h_new, float *h_rk, float *u, float *v, parameters p) {
    std::vector<float> flx(p.n()), fly(p.n());
    float max_height;

    // 3rd-order 3-stage TVD Runge-Kutta method
    // 1st stage h_old --> h_new
    height_flux(flx.data(), fly.data(), h_old, u, v, p);
    max_height = height_integral(h_new, h_old, h_old, flx.data(), fly.data(), p, 0.f, 1.f);
    boundary_height(h_new, p);

    // 2nd stage h_new --> h_rk
    height_flux(flx.data(), fly.data(), h_new, u, v, p);
    max_height =
        height_integral(h_rk, h_new, h_old, flx.data(), fly.data(), p, 3.f / 4.f, 1.f / 4.f);
    boundary_height(h_rk, p);

    // 3rd stage h_rk --> h_new
    height_flux(flx.data(), fly.data(), h_rk, u, v, p);
    max_height =
        height_integral(h_new, h_rk, h_old, flx.data(), fly.data(), p, 1.f / 3.f, 2.f / 3.f);
    boundary_height(h_new, p);

    return max_height;
}

float momentum_stencil(float *u_new, float *v_new, float *u_old, float *v_old, float *u_n,
                       float *v_n, float *h, parameters p, float c0, float c1) {
    grid g{.x_begin = p.halo / 2,
           .x_end = p.nx + p.halo / 2,
           .y_begin = p.halo / 2,
           .y_end = p.ny + p.halo / 2};

    auto xs = std::views::iota(g.x_begin, g.x_end);
    auto ys = std::views::iota(g.y_begin, g.y_end);
    auto ids = std::views::cartesian_product(xs, ys);

    float max_vel = std::transform_reduce(
        std::execution::par, ids.begin(), ids.end(), std::numeric_limits<float>::min(),
        [](auto a, auto b) { return std::max(a, b); },
        [u_new, v_new, u_old, v_old, u_n, v_n, h, p, c0, c1](auto gid) {
            auto idx = [=](auto i, auto j) { return j * (p.nx + p.halo) + i; };
            auto [i, j] = gid;

            float u_adv, v_adv, adv_term, grad_term;
            int upwind;

            // update u
            u_adv = u_old[idx(i, j)];
            v_adv = 0.25f * (v_old[idx(i - 1, j)] + v_old[idx(i, j)] + v_old[idx(i - 1, j + 1)] +
                             v_old[idx(i, j + 1)]);

            adv_term = 0;
            upwind = u_adv < 0 ? 1 : -1;
            adv_term += u_adv * scheme::HJ_WENO3(u_old[idx(i - upwind, j)], u_old[idx(i, j)],
                                                 u_old[idx(i + upwind, j)],
                                                 u_old[idx(i + 2 * upwind, j)], u_adv, p.dx);
            upwind = v_adv < 0 ? 1 : -1;
            adv_term += v_adv * scheme::HJ_WENO3(u_old[idx(i, j - upwind)], u_old[idx(i, j)],
                                                 u_old[idx(i, j + upwind)],
                                                 u_old[idx(i, j + 2 * upwind)], v_adv, p.dx);
            grad_term = p.gravity * (h[idx(i, j)] - h[idx(i - 1, j)]) / p.dx;

            u_new[idx(i, j)] =
                c0 * u_n[idx(i, j)] + c1 * (u_old[idx(i, j)] - (adv_term + grad_term) * p.dt);

            // update v
            u_adv = 0.25f * (u_old[idx(i, j - 1)] + u_old[idx(i, j)] + u_old[idx(i + 1, j - 1)] +
                             u_old[idx(i + 1, j)]);
            v_adv = v_old[idx(i, j)];

            adv_term = 0;
            upwind = u_adv < 0 ? 1 : -1;
            adv_term += u_adv * scheme::HJ_WENO3(v_old[idx(i - upwind, j)], v_old[idx(i, j)],
                                                 v_old[idx(i + upwind, j)],
                                                 v_old[idx(i + 2 * upwind, j)], u_adv, p.dx);
            upwind = v_adv < 0 ? 1 : -1;
            adv_term += v_adv * scheme::HJ_WENO3(v_old[idx(i, j - upwind)], v_old[idx(i, j)],
                                                 v_old[idx(i, j + upwind)],
                                                 v_old[idx(i, j + 2 * upwind)], v_adv, p.dx);
            grad_term = p.gravity * (h[idx(i, j)] - h[idx(i, j - 1)]) / p.dx;

            v_new[idx(i, j)] =
                c0 * v_n[idx(i, j)] + c1 * (v_old[idx(i, j)] - (adv_term + grad_term) * p.dt);

            return std::abs(u_new[idx(i, j)]) + std::abs(v_new[idx(i, j)]);
        });

    return max_vel;
}

float solve_momentum(float *u_old, float *v_old, float *u_new, float *v_new, float *u_rk,
                     float *v_rk, float *h, parameters p) {
    float max_vel;

    // 3rd-order 3-stage TVD Runge-Kutta method
    // 1st stage u_old, v_old --> u_new, v_new
    max_vel = momentum_stencil(u_new, v_new, u_old, v_old, u_old, v_old, h, p, 0.f, 1.f);
    boundary_velocity(u_new, v_new, p);

    // 2nd stage u_new, v_new --> u_rk, v_rk
    max_vel = momentum_stencil(u_rk, v_rk, u_new, v_new, u_old, v_old, h, p, 3.f / 4.f, 1.f / 4.f);
    boundary_velocity(u_rk, v_rk, p);

    // 3rd stage u_rk, v_rk --> u_new, v_new
    max_vel = momentum_stencil(u_new, v_new, u_rk, v_rk, u_old, v_old, h, p, 1.f / 3.f, 2.f / 3.f);
    boundary_velocity(u_new, v_new, p);

    return max_vel;
}

void write_to_csv(float *u, parameters p, std::string filename);

int main(int argc, char *argv[]) {
    // Parse CLI parameters
    parameters p(argc, argv);

    const float CFL = 0.3;
    const float init_h = 0.5;

    // Allocate memory
    std::vector<float> h_old(p.n()), h_new(p.n()), h_rk(p.n());
    std::vector<float> u_old(p.n()), u_new(p.n()), u_rk(p.n());
    std::vector<float> v_old(p.n()), v_new(p.n()), v_rk(p.n());

    initial_condition(h_old.data(), u_old.data(), v_old.data(), p, init_h);
    p.dt = CFL * p.dx / std::sqrt(p.gravity * init_h);

    float Tsim = 0.f;
    uint output = 0;

    // main loop
    for (uint i = 0; Tsim <= p.Tend; ++i) {
        float h_max =
            solve_height(h_old.data(), h_new.data(), h_rk.data(), u_old.data(), v_old.data(), p);
        float v_max = solve_momentum(u_old.data(), v_old.data(), u_new.data(), v_new.data(),
                                     u_rk.data(), v_rk.data(), h_old.data(), p);

        // output data
        if (uint(Tsim / p.Tout()) >= output) {
            std::cout << "Time = " << Tsim << " [sec], steps = " << i << ", max vel = " << v_max
                      << ", max height = " << h_max << std::endl;

            std::string filename = "height_" + std::to_string(output) + ".csv";
            write_to_csv(h_old.data(), p, filename);

            output++;
        }

        std::swap(h_new, h_old);
        std::swap(u_new, u_old);
        std::swap(v_new, v_old);

        Tsim += p.dt;

        p.dt = CFL * p.dx / (v_max + std::sqrt(p.gravity * h_max));
    }

    return 0;
}

void write_to_csv(float *h, parameters p, std::string filename) {
    auto idx = [=](auto x, auto y) { return y * (p.nx + p.halo) + x; };

    std::ofstream file(filename);

    for (int j = 0; j < p.ny; j++) {
        for (int i = 0; i < p.nx; i++) {
            if (i != 0) {
                file << ",";
            }
            file << h[idx(i + p.halo / 2, j + p.halo / 2)];
        }
        file << std::endl;
    }

    file.close();
}