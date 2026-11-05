// phyz_convex.cu — phyz_contact's convex contact solve, f64, one thread per
// problem.
//
// A transcription of `phyz_contact::convex::solve_contacts_warm` (forward
// path, preconditioner off — its default) for `ContactCoupling::Full` and
// `BlockDiagonal`: the normal warm-up sweeps, the full staged sweeps, then
// active-set Newton (`classify_impulses` -> `kkt_matrix` -> `solve_dense` ->
// staged clamp) with the 3-step line search, PGS blocks and the stagnation
// exit. Same expressions in the same order, so with FMA contraction off on
// both sides (NVRTC `--fmad=false`; the host build passes
// `-ffp-contract=off`) the result is the CPU solver's to the bit.
//
// Built two ways, like phyz_kernels.cu: NVRTC at runtime for the device, and
// as host C++ under `cuda-host` for tests without a GPU.
//
// Layout, per problem p (strides are the batch's NMAX contacts):
//   ncont[p]                       contact count n <= nmax
//   a    [p][3nmax][3nmax]          Delassus, row-major in its leading 3n x 3n
//   b    [p][3nmax]                 free velocity
//   rows [p][nmax][ROW_STRIDE]      mu, bias, effective impedance
//   f    [p][3nmax]                 in: warm-start seed; out: impulses
//   stat [p][2]                     out: iterations, residual
//   scratch [p][SCRATCH(nmax)]      blocks, KKT matrix, rhs, trial, candidate

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define CVX_ON_DEVICE 1
#define CVX_DEV __device__ __forceinline__
#else
#define CVX_ON_DEVICE 0
#define CVX_DEV static inline
#include <math.h>
#pragma clang fp contract(off)
#endif

typedef unsigned int u32;
typedef unsigned long long u64;

#define ROW_STRIDE 3u
#define WARMUP_SWEEPS 8u
#define NEWTON_ATTEMPTS 24u
#define LINE_SEARCH_STEPS 3u
#define PGS_BLOCK 4u
#define STALL_RATIO 0.99
#define STALL_BLOCKS 3u
#define COUPLING_FULL 0u
#define COUPLING_BLOCK_DIAGONAL 2u

typedef struct {
    double regularization, tolerance, relaxation;
    u32 max_iterations, newton, mujoco_compat, coupling, no_stall_exit;
} cvx_cfg;

CVX_DEV double dmax(double a, double b) { return a > b ? a : b; }  // f64::max (no NaN inputs)
CVX_DEV double dmin(double a, double b) { return a < b ? a : b; }
CVX_DEV double dabs(double a) { return fabs(a); }

// regularization_diag
CVX_DEV void reg_diag(const double* a, u32 dim, const double* row, u32 c, const cvx_cfg* cfg, double out[3]) {
    u32 base = 3u * c;
    double d = row[2];
    d = d < 1e-6 ? 1e-6 : (d > 1.0 ? 1.0 : d);
    double scale = (1.0 - d) / d;
    double a_nn = a[(u64)base * dim + base];
    double normal = dmax(scale * a_nn, cfg->regularization);
    double tangent = cfg->mujoco_compat ? normal : cfg->regularization;
    out[0] = normal; out[1] = tangent; out[2] = tangent;
}

// disc_block_step: minimise the 2x2 tangential block on the disc |t| = limit.
CVX_DEV void disc_block_step(const double m[2][2], const double r[2], double limit, double t_out[2]) {
    double k = 0.0;
    double t[2];
#define CVX_SOLVE(KK, R0, R1, O)                                         \
    {                                                                    \
        double aa = m[0][0] + (KK), bb = m[0][1], cc = m[1][0], dd = m[1][1] + (KK); \
        double det = aa * dd - bb * cc;                                  \
        double r0 = (R0), r1 = (R1);                                     \
        O[0] = (dd * r0 - bb * r1) / det;                                \
        O[1] = (aa * r1 - cc * r0) / det;                                \
    }
    CVX_SOLVE(k, -r[0], -r[1], t);
    for (u32 it = 0; it < 60u; it++) {
        double nrm = sqrt(t[0] * t[0] + t[1] * t[1]);
        if (dabs(nrm - limit) <= 1e-14 * (1.0 + limit)) break;
        double st[2];
        CVX_SOLVE(k, t[0], t[1], st);
        double dn = -(t[0] * st[0] + t[1] * st[1]) / nrm;
        double phi = 1.0 / nrm - 1.0 / limit;
        double dphi = -dn / (nrm * nrm);
        double next = dmax(k - phi / dphi, 0.0);
        if (next == k) break;
        k = next;
        CVX_SOLVE(k, -r[0], -r[1], t);
    }
#undef CVX_SOLVE
    double nrm = sqrt(t[0] * t[0] + t[1] * t[1]);
    double s = limit / nrm;
    t_out[0] = t[0] * s;
    t_out[1] = t[1] * s;
}

// sweep (primal). Returns max_move.
CVX_DEV double sweep(u32 n, const double* a, const double* b, const double* rows,
                     const double* blocks, double* f, u32 normals_only, const cvx_cfg* cfg) {
    u32 dim = 3u * n;
    double max_move = 0.0;
    for (u32 c = 0; c < n; c++) {
        u32 base = 3u * c;
        double r[3];
        for (u32 row = 0; row < 3u; row++) {
            double acc = b[base + row];
            if (cfg->coupling != COUPLING_BLOCK_DIAGONAL) {
                const double* ar = a + (u64)(base + row) * dim;
                for (u32 k = 0; k < n; k++) {
                    if (k == c) continue;
                    u32 kb = 3u * k;
                    acc += ar[kb] * f[kb] + ar[kb + 1u] * f[kb + 1u] + ar[kb + 2u] * f[kb + 2u];
                }
            }
            r[row] = acc;
        }
        const double* rw = rows + (u64)c * ROW_STRIDE;
        double mu = rw[0], bias = rw[1];
        const double* bl = blocks + 9u * c;
        double fy = f[base + 1u], fz = f[base + 2u];
        double a_nn = bl[0];
        double a_nu = bl[1], a_nw = bl[2];
        double unclamped = a_nn > 0.0 ? (bias - r[0] - a_nu * fy - a_nw * fz) / a_nn : 0.0;
        double f_n = a_nn > 0.0 ? dmax(unclamped, 0.0) : 0.0;
        double a_un = bl[3], a_wn = bl[6];
        double r_u = r[1] + a_un * f_n;
        double r_w = r[2] + a_wn * f_n;
        double m00 = bl[4], m01 = bl[5], m10 = bl[7], m11 = bl[8];
        double det = m00 * m11 - m01 * m10;
        double t_u, t_w;
        if (!normals_only) {
            if (dabs(det) > 1e-18) {
                t_u = -(m11 * r_u - m01 * r_w) / det;
                t_w = -(m00 * r_w - m10 * r_u) / det;
            } else { t_u = 0.0; t_w = 0.0; }
        } else { t_u = fy; t_w = fz; }
        double limit = mu * f_n;
        double t_norm = sqrt(t_u * t_u + t_w * t_w);
        if (t_norm > limit && limit > 0.0 && !normals_only && dabs(det) > 1e-18) {
            double m[2][2] = {{m00, m01}, {m10, m11}};
            double rr[2] = {r_u, r_w};
            double t[2];
            disc_block_step(m, rr, limit, t);
            t_u = t[0]; t_w = t[1];
        } else if (t_norm > limit) {
            if (t_norm > 0.0) {
                double s = limit / t_norm;
                t_u *= s; t_w *= s;
            } else { t_u = 0.0; t_w = 0.0; }
        }
        double tgt[3] = {f_n, t_u, t_w};
        double mv2 = 0.0;
        for (u32 i = 0; i < 3u; i++) {
            double cur = f[base + i];
            double next = cur + (tgt[i] - cur) * cfg->relaxation;
            double d = next - cur;
            mv2 += d * d;
            f[base + i] = next;
        }
        max_move = dmax(max_move, sqrt(mv2));
    }
    return max_move;
}

// newton_step_diff (primal). Writes the candidate into `cand`; returns 0 when
// the KKT solve is singular or non-finite (`None`).
CVX_DEV u32 newton_step(u32 n, const double* a, const double* b, const double* rows,
                        const double* f, const cvx_cfg* cfg, double* kk, double* rhs, double* cand) {
    u32 dim = 3u * n;
    for (u64 i = 0; i < (u64)dim * dim; i++) kk[i] = 0.0;
    for (u32 i = 0; i < dim; i++) rhs[i] = 0.0;
    const double tol = 1e-7;
    for (u32 c = 0; c < n; c++) {
        u32 base = 3u * c;
        const double* rw = rows + (u64)c * ROW_STRIDE;
        double fx = f[base], fy = f[base + 1u], fz = f[base + 2u];
        // classify_impulses: 0 separating, 1 sticking, 2 sliding
        u32 regime;
        if (fx <= tol) regime = 0u;
        else {
            double ft = sqrt(fy * fy + fz * fz);
            double limit = rw[0] * fx;
            regime = (limit <= 0.0 || ft >= limit - tol * dmax(fx, 1.0)) ? 2u : 1u;
        }
        if (regime == 0u) {
            for (u32 r = 0; r < 3u; r++) kk[(u64)(base + r) * dim + base + r] = 1.0;
        } else if (regime == 1u) {
            double reg[3];
            reg_diag(a, dim, rw, c, cfg, reg);
            for (u32 r = 0; r < 3u; r++) {
                for (u32 col = 0; col < dim; col++)
                    kk[(u64)(base + r) * dim + col] = a[(u64)(base + r) * dim + col];
                kk[(u64)(base + r) * dim + base + r] += reg[r];
            }
            rhs[base] = rw[1] - b[base];
            rhs[base + 1u] = -b[base + 1u];
            rhs[base + 2u] = -b[base + 2u];
        } else {
            double reg[3];
            reg_diag(a, dim, rw, c, cfg, reg);
            for (u32 col = 0; col < dim; col++) kk[(u64)base * dim + col] = a[(u64)base * dim + col];
            kk[(u64)base * dim + base] += reg[0];
            double ft = sqrt(fy * fy + fz * fz);
            double tu = 0.0, tw = 0.0;
            if (ft > 1e-14) { tu = fy / ft; tw = fz / ft; }
            double mu = rw[0];
            kk[(u64)(base + 1u) * dim + base + 1u] = 1.0;
            kk[(u64)(base + 1u) * dim + base] = -mu * tu;
            kk[(u64)(base + 2u) * dim + base + 2u] = 1.0;
            kk[(u64)(base + 2u) * dim + base] = -mu * tw;
            rhs[base] = rw[1] - b[base];
        }
    }
    // solve_dense (Gauss-Jordan, partial pivoting), ncols = 1
    for (u32 col = 0; col < dim; col++) {
        u32 pivot = col;
        for (u32 r = col + 1u; r < dim; r++)
            if (dabs(kk[(u64)r * dim + col]) > dabs(kk[(u64)pivot * dim + col])) pivot = r;
        if (dabs(kk[(u64)pivot * dim + col]) < 1e-14) return 0u;
        if (pivot != col) {
            for (u32 k = 0; k < dim; k++) {
                double t = kk[(u64)col * dim + k];
                kk[(u64)col * dim + k] = kk[(u64)pivot * dim + k];
                kk[(u64)pivot * dim + k] = t;
            }
            double t = rhs[col]; rhs[col] = rhs[pivot]; rhs[pivot] = t;
        }
        double d = kk[(u64)col * dim + col];
        for (u32 k = 0; k < dim; k++) kk[(u64)col * dim + k] /= d;
        rhs[col] /= d;
        for (u32 r = 0; r < dim; r++) {
            if (r == col) continue;
            double factor = kk[(u64)r * dim + col];
            if (factor == 0.0) continue;
            for (u32 k = 0; k < dim; k++) kk[(u64)r * dim + k] -= factor * kk[(u64)col * dim + k];
            rhs[r] -= factor * rhs[col];
        }
    }
    for (u32 i = 0; i < dim; i++) if (!isfinite(rhs[i])) return 0u;
    for (u32 c = 0; c < n; c++) {
        u32 base = 3u * c;
        double f_n = dmax(rhs[base], 0.0);
        double t_u = rhs[base + 1u], t_w = rhs[base + 2u];
        double limit = rows[(u64)c * ROW_STRIDE] * f_n;
        double t_norm = sqrt(t_u * t_u + t_w * t_w);
        if (t_norm > limit) {
            double s = t_norm > 0.0 ? limit / t_norm : 0.0;
            t_u *= s; t_w *= s;
        }
        cand[base] = f_n; cand[base + 1u] = t_u; cand[base + 2u] = t_w;
    }
    return 1u;
}

CVX_DEV u64 cvx_scratch_len(u32 nmax) {
    u64 dim = 3ull * nmax;
    return 9ull * nmax + dim * dim + 3ull * dim;  // blocks, kkt, rhs, trial, cand
}

CVX_DEV void convex_thread(u32 p, u32 nprob, u32 nmax, cvx_cfg cfg, const u32* ncont,
                           const double* a_all, const double* b_all, const double* rows_all,
                           double* f_all, double* stat, double* scratch_all) {
    if (p >= nprob) return;
    u32 n = ncont[p];
    u64 dmax3 = 3ull * nmax;
    const double* a = a_all + (u64)p * dmax3 * dmax3;
    const double* b = b_all + (u64)p * dmax3;
    const double* rows = rows_all + (u64)p * nmax * ROW_STRIDE;
    double* f = f_all + (u64)p * dmax3;
    double* sc = scratch_all + (u64)p * cvx_scratch_len(nmax);
    // The Delassus is packed at its own leading dimension 3n (so `a` reads
    // exactly like the CPU's dim x dim array).
    u32 dim = 3u * n;
    double* blocks = sc;
    double* kk = blocks + 9ull * nmax;
    double* rhs = kk + dmax3 * dmax3;
    double* trial = rhs + dmax3;
    double* cand = trial + dmax3;
    if (n == 0u) { stat[2ull * p] = 0.0; stat[2ull * p + 1] = 0.0; return; }

    for (u32 c = 0; c < n; c++) {
        double reg[3];
        reg_diag(a, dim, rows + (u64)c * ROW_STRIDE, c, &cfg, reg);
        u32 base = 3u * c;
        for (u32 r = 0; r < 3u; r++)
            for (u32 col = 0; col < 3u; col++) {
                double e = a[(u64)(base + r) * dim + base + col];
                if (r == col) e += reg[r];
                blocks[9u * c + 3u * r + col] = e;
            }
    }

    u32 iterations = 0;
    u32 warm = WARMUP_SWEEPS < cfg.max_iterations ? WARMUP_SWEEPS : cfg.max_iterations;
    for (u32 i = 0; i < warm; i++) {
        iterations++;
        if (sweep(n, a, b, rows, blocks, f, 1u, &cfg) < cfg.tolerance) break;
    }
    double residual = 1.0 / 0.0;  // +inf; NVRTC has no math.h
    u32 cap = cfg.max_iterations < 2u * WARMUP_SWEEPS ? cfg.max_iterations : 2u * WARMUP_SWEEPS;
    while (iterations < cap) {
        iterations++;
        residual = sweep(n, a, b, rows, blocks, f, 0u, &cfg);
        if (residual < cfg.tolerance) {
            stat[2ull * p] = (double)iterations; stat[2ull * p + 1] = residual;
            return;
        }
    }

    u32 newton_solves = 0, stalls = 0;
    while (iterations < cfg.max_iterations) {
        double entry_residual = residual;
        u32 have = 0u;
        if (cfg.newton && newton_solves < NEWTON_ATTEMPTS)
            have = newton_step(n, a, b, rows, f, &cfg, kk, rhs, cand);
        if (have) {
            newton_solves++;
            iterations++;
            double alpha = 1.0;
            for (u32 ls = 0; ls < LINE_SEARCH_STEPS; ls++) {
                for (u32 i = 0; i < dim; i++) trial[i] = f[i] + (cand[i] - f[i]) * alpha;
                iterations++;
                double tr = sweep(n, a, b, rows, blocks, trial, 0u, &cfg);
                if (tr < residual) {
                    residual = tr;
                    for (u32 i = 0; i < dim; i++) f[i] = trial[i];
                    break;
                }
                alpha *= 0.5;
            }
            if (residual < cfg.tolerance) break;
        }
        for (u32 i = 0; i < PGS_BLOCK; i++) {
            if (iterations >= cfg.max_iterations) break;
            iterations++;
            residual = sweep(n, a, b, rows, blocks, f, 0u, &cfg);
            if (residual < cfg.tolerance) break;
        }
        if (residual < cfg.tolerance) break;
        if (residual > entry_residual * STALL_RATIO) {
            stalls++;
            if (stalls >= STALL_BLOCKS && !cfg.no_stall_exit) break;
        } else {
            stalls = 0;
        }
    }
    stat[2ull * p] = (double)iterations;
    stat[2ull * p + 1] = residual;
}

#if CVX_ON_DEVICE
extern "C" __global__ void phyz_convex_solve(u32 nprob, u32 nmax,
                                             double regularization, double tolerance, double relaxation,
                                             u32 max_iterations, u32 newton, u32 mujoco_compat,
                                             u32 coupling, u32 no_stall_exit,
                                             const u32* ncont, const double* a, const double* b,
                                             const double* rows, double* f, double* stat,
                                             double* scratch) {
    cvx_cfg cfg = {regularization, tolerance, relaxation, max_iterations, newton,
                   mujoco_compat, coupling, no_stall_exit};
    convex_thread(blockIdx.x * blockDim.x + threadIdx.x, nprob, nmax, cfg, ncont, a, b, rows, f,
                  stat, scratch);
}
#else
extern "C" void phyz_host_convex_solve(u32 nprob, u32 nmax,
                                       double regularization, double tolerance, double relaxation,
                                       u32 max_iterations, u32 newton, u32 mujoco_compat,
                                       u32 coupling, u32 no_stall_exit,
                                       const u32* ncont, const double* a, const double* b,
                                       const double* rows, double* f, double* stat,
                                       double* scratch) {
    cvx_cfg cfg = {regularization, tolerance, relaxation, max_iterations, newton,
                   mujoco_compat, coupling, no_stall_exit};
    for (u32 p = 0; p < nprob; p++)
        convex_thread(p, nprob, nmax, cfg, ncont, a, b, rows, f, stat, scratch);
}
#endif
