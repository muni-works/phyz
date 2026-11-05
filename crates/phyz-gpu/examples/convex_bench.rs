//! Device convex-solve parity and cost on recorded problems.
//!
//! ```text
//! cargo run --release -p phyz-gpu --features cuda --example convex_bench -- DUMP.bin [copies...]
//! ```
//!
//! `DUMP.bin` is a `CVX1` file (ipse's `k1_contact_dump`: problem, seed, the
//! CPU solver's impulses / iterations / residual). Two checks:
//!
//! 1. **Parity** — every problem solved once on the device, compared to the
//!    CPU's answer bit for bit (and, failing that, the max relative error).
//! 2. **Cost** — the recorded set tiled to each batch size in `copies`
//!    (default 1024 4096 16384), timed per kernel launch, reported as µs per
//!    solve and solves/s; plus the CPU solver's own µs/solve on one thread.

use std::time::Instant;

use phyz_contact::{ContactCoupling, ContactProblem, ContactRow, ContactSolverConfig, solve_contacts_warm};
use phyz_gpu::cuda::convex::{ConvexBatch, ConvexCfg, ConvexRow, CudaConvex};
use phyz_math::Vec3;

struct Rec {
    cfg: ConvexCfg,
    n: usize,
    a: Vec<f64>,
    b: Vec<f64>,
    rows: Vec<ConvexRow>,
    seed: Vec<[f64; 3]>,
    want: Vec<[f64; 3]>,
    iters: u32,
    residual: f64,
}

fn read(path: &str) -> Vec<Rec> {
    let d = std::fs::read(path).expect("read dump");
    assert_eq!(&d[..4], b"CVX1", "not a CVX1 dump");
    let mut o = 4usize;
    let u32_ = |o: &mut usize| {
        let v = u32::from_le_bytes(d[*o..*o + 4].try_into().unwrap());
        *o += 4;
        v
    };
    let f64_ = |o: &mut usize| {
        let v = f64::from_le_bytes(d[*o..*o + 8].try_into().unwrap());
        *o += 8;
        v
    };
    let count = u32_(&mut o);
    (0..count)
        .map(|_| {
            let n = u32_(&mut o) as usize;
            let (reg, tol, rel) = (f64_(&mut o), f64_(&mut o), f64_(&mut o));
            let maxit = u32_(&mut o);
            let fl = &d[o..o + 4];
            let cfg = ConvexCfg {
                regularization: reg,
                tolerance: tol,
                relaxation: rel,
                max_iterations: maxit,
                newton: fl[0] != 0,
                mujoco_compat: fl[1] != 0,
                block_diagonal: fl[2] != 0,
                no_stall_exit: fl[3] != 0,
            };
            o += 4;
            let mut vec = |k: usize, o: &mut usize| (0..k).map(|_| f64_(o)).collect::<Vec<f64>>();
            let a = vec(9 * n * n, &mut o);
            let b = vec(3 * n, &mut o);
            let r = vec(3 * n, &mut o);
            let s = vec(3 * n, &mut o);
            let w = vec(3 * n, &mut o);
            let iters = u32_(&mut o);
            let residual = f64_(&mut o);
            let tri = |v: &[f64]| v.chunks(3).map(|c| [c[0], c[1], c[2]]).collect::<Vec<_>>();
            Rec {
                cfg,
                n,
                a,
                b,
                rows: r.chunks(3).map(|c| ConvexRow { mu: c[0], bias: c[1], impedance: c[2] }).collect(),
                seed: tri(&s),
                want: tri(&w),
                iters,
                residual,
            }
        })
        .collect()
}

fn batch(recs: &[Rec], copies: usize) -> ConvexBatch {
    let nmax = recs.iter().map(|r| r.n).max().unwrap_or(1);
    let mut b = ConvexBatch::new(nmax);
    for i in 0..copies {
        let r = &recs[i % recs.len()];
        b.push(&r.a, &r.b, &r.rows, &r.seed);
    }
    b
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let recs = read(args.get(1).expect("usage: convex_bench DUMP.bin [copies...]"));
    let copies: Vec<usize> = if args.len() > 2 {
        args[2..].iter().map(|s| s.parse().unwrap()).collect()
    } else {
        vec![1024, 4096, 16384]
    };
    let cfg = recs[0].cfg;
    let nmax = recs.iter().map(|r| r.n).max().unwrap();
    let mean_n = recs.iter().map(|r| r.n).sum::<usize>() as f64 / recs.len() as f64;
    let newton = recs.iter().filter(|r| r.iters > 16).count();
    println!(
        "{} problems, n mean {mean_n:.1} max {nmax}, {newton} reached Newton, tol {:.0e}",
        recs.len(),
        cfg.tolerance
    );
    let dev = CudaConvex::new(0).expect("CUDA");

    // 1. Parity.
    let mut b = batch(&recs, recs.len());
    let (st, _) = dev.solve(&cfg, &mut b, 64).expect("solve");
    let (mut exact, mut worst, mut it_ok) = (0usize, 0.0f64, 0usize);
    for (p, r) in recs.iter().enumerate() {
        let got = b.impulses(p);
        let scale = r.want.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-300);
        let mut bits = true;
        for (g, w) in got.iter().flatten().zip(r.want.iter().flatten()) {
            if g.to_bits() != w.to_bits() {
                bits = false;
                worst = worst.max((g - w).abs() / scale);
            }
        }
        if bits && st[p].1.to_bits() == r.residual.to_bits() {
            exact += 1;
        }
        if st[p].0 == r.iters {
            it_ok += 1;
        }
    }
    println!(
        "parity: {exact}/{} bit-identical (impulses + residual), {it_ok} same iteration count, worst rel err {worst:.2e}",
        recs.len()
    );

    // CPU reference cost, one thread.
    let cc = ContactSolverConfig {
        regularization: cfg.regularization,
        tolerance: cfg.tolerance,
        relaxation: cfg.relaxation,
        max_iterations: cfg.max_iterations as usize,
        newton: cfg.newton,
        mujoco_compat: cfg.mujoco_compat,
        coupling: if cfg.block_diagonal { ContactCoupling::BlockDiagonal } else { ContactCoupling::Full },
        no_stall_exit: cfg.no_stall_exit,
        ..ContactSolverConfig::simulation()
    };
    let probs: Vec<(ContactProblem, Vec<Vec3>)> = recs
        .iter()
        .map(|r| {
            let rows = r
                .rows
                .iter()
                .map(|x| ContactRow { mu: x.mu, bias: x.bias, impedance: x.impedance, ..ContactRow::default() })
                .collect();
            (
                ContactProblem { n: r.n, delassus: r.a.clone(), free_velocity: r.b.clone(), rows, bodies: Vec::new() },
                r.seed.iter().map(|s| Vec3::new(s[0], s[1], s[2])).collect(),
            )
        })
        .collect();
    let t = Instant::now();
    for (p, s) in &probs {
        std::hint::black_box(solve_contacts_warm(p, &cc, s));
    }
    let cpu = t.elapsed().as_secs_f64() / probs.len() as f64;
    println!("cpu solve_contacts_warm: {:.2} us/solve (1 thread)", cpu * 1e6);

    // 2. Cost.
    for &c in &copies {
        for block in [32u32, 64, 128] {
            let mut b = batch(&recs, c);
            let _ = dev.solve(&cfg, &mut b, block).expect("warm-up");
            let mut best = f64::MAX;
            for _ in 0..3 {
                let mut b = batch(&recs, c);
                let (_, s) = dev.solve(&cfg, &mut b, block).expect("solve");
                best = best.min(s);
            }
            println!(
                "device {c:>6} solves, block {block:>3}: {:8.3} ms/launch  {:7.3} us/solve  {:.2e} solves/s  (= {:.0} CPU threads)",
                best * 1e3,
                best / c as f64 * 1e6,
                c as f64 / best,
                cpu * c as f64 / best
            );
        }
    }
}
