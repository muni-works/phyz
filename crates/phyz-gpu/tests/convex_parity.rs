//! The device convex solve (`cuda/phyz_convex.cu`, run here as host C++) is
//! `phyz_contact::solve_contacts_warm`, bit for bit.
//!
//! Fixtures are real trajectories through phyz's own CPU pipeline — FK,
//! `find_ground_contacts_model`, ABA, `assemble`, the warm-started solve,
//! velocity update, integrate — and every contact problem met on the way is
//! recorded with the seed the CPU used and the solution it got. The kernel
//! then solves the same problems from the same seeds. Impulses, iteration
//! counts and residuals must agree to the bit: the kernel is a transcription,
//! not a model, and FMA contraction is off on both sides.
#![cfg(feature = "cuda-host")]

use phyz_contact::{
    ContactCoupling, ContactMaterial, ContactProblem, ContactSolverConfig, assemble,
    find_ground_contacts_model, solve_contacts_warm,
};
use phyz_gpu::cuda::convex::{ConvexBatch, ConvexCfg, ConvexRow, solve_host};
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, Joint, JointType, Model, ModelBuilder};

pub struct Recorded {
    pub problem: ContactProblem,
    pub seed: Vec<Vec3>,
    pub impulses: Vec<Vec3>,
    pub iterations: usize,
    pub residual: f64,
}

fn box_inertia(mass: f64, h: Vec3) -> SpatialInertia {
    let (lx, ly, lz) = (2.0 * h.x, 2.0 * h.y, 2.0 * h.z);
    SpatialInertia::new(
        mass,
        Vec3::zeros(),
        Mat3::from_diagonal(&Vec3::new(
            mass / 12.0 * (ly * ly + lz * lz),
            mass / 12.0 * (lx * lx + lz * lz),
            mass / 12.0 * (lx * lx + ly * ly),
        )),
    )
}

fn dropped_box() -> Model {
    let h = Vec3::new(0.1, 0.07, 0.05);
    let mut m = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(0.001)
        .add_body(
            "box",
            -1,
            Joint::free(SpatialTransform::identity()),
            box_inertia(1.5, h),
        )
        .build();
    m.bodies[0].geometry = Some(Geometry::Box { half_extents: h });
    m
}

/// A box "foot" under a hinged shank: the Delassus couples through a chain.
fn foot_chain() -> Model {
    let foot = Vec3::new(0.11, 0.05, 0.02);
    let shank = Vec3::new(0.03, 0.03, 0.15);
    let mut m = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(0.002)
        .add_body(
            "shank",
            -1,
            Joint::free(SpatialTransform::identity()),
            box_inertia(4.0, shank),
        )
        .add_body(
            "foot",
            0,
            Joint {
                joint_type: JointType::Revolute,
                parent_to_joint: SpatialTransform::new(
                    Mat3::identity(),
                    Vec3::new(0.0, 0.0, -0.17),
                ),
                axis: Vec3::new(0.0, 1.0, 0.0),
                damping: 0.5,
                armature: 0.01,
                ..Joint::default()
            },
            box_inertia(0.6, foot),
        )
        .build();
    m.bodies[1].geometry = Some(Geometry::Box { half_extents: foot });
    m
}

/// Run `steps` of the CPU pipeline from `init` and record every problem.
pub fn record(model: &Model, init: phyz_model::State, steps: usize, mu: f64) -> Vec<Recorded> {
    let cfg = ContactSolverConfig::simulation();
    let mat = ContactMaterial {
        friction: mu,
        ..ContactMaterial::default()
    };
    let mats = vec![mat.clone(); model.bodies.len()];
    let mut s = init;
    let mut warm: Vec<Vec3> = Vec::new();
    let mut out = Vec::new();
    for _ in 0..steps {
        let (xf, _) = phyz_rigid::forward_kinematics(model, &s);
        s.body_xform = xf;
        let contacts = find_ground_contacts_model(model, &s, 0.0, mat.margin);
        let mut qdd = phyz_rigid::aba(model, &s);
        let v0 = s.v.clone();
        phyz_rigid::strip_free_joint_coriolis(model, v0.as_slice(), qdd.as_mut_slice());
        let free = &s.v + &(&qdd * model.dt);
        if contacts.is_empty() {
            s.v = free;
            warm.clear();
        } else {
            let asm = assemble(model, &s, &contacts, &mats, &free, model.dt, &cfg);
            // Seed: last step's impulses when the contact count held (the
            // stepper's cache keys by contact; a count match is the common case).
            let seed = if warm.len() == contacts.len() {
                warm.clone()
            } else {
                Vec::new()
            };
            let sol = solve_contacts_warm(&asm.problem, &cfg, &seed);
            s.v = &free + &asm.velocity_delta(&sol.impulses);
            warm = sol.impulses.clone();
            out.push(Recorded {
                problem: asm.problem,
                seed,
                impulses: sol.impulses,
                iterations: sol.iterations,
                residual: sol.residual,
            });
        }
        phyz_rigid::rotate_free_joint_velocities(
            model,
            v0.as_slice(),
            s.v.as_mut_slice(),
            model.dt,
        );
        let v = s.v.clone();
        phyz_rigid::integrate_configuration(model, s.q.as_mut_slice(), v.as_slice(), model.dt);
        s.time += model.dt;
    }
    out
}

pub fn cfg_of(c: &ContactSolverConfig) -> ConvexCfg {
    ConvexCfg {
        regularization: c.regularization,
        tolerance: c.tolerance,
        relaxation: c.relaxation,
        max_iterations: c.max_iterations as u32,
        newton: c.newton,
        mujoco_compat: c.mujoco_compat,
        block_diagonal: c.coupling == ContactCoupling::BlockDiagonal,
        no_stall_exit: c.no_stall_exit,
    }
}

pub fn batch_of(recs: &[Recorded]) -> ConvexBatch {
    let nmax = recs.iter().map(|r| r.problem.n).max().unwrap_or(1);
    let mut b = ConvexBatch::new(nmax);
    for r in recs {
        let rows: Vec<ConvexRow> = r
            .problem
            .rows
            .iter()
            .map(|row| ConvexRow {
                mu: row.mu,
                bias: row.bias,
                impedance: row.effective_impedance(),
            })
            .collect();
        let seed: Vec<[f64; 3]> = r.seed.iter().map(|v| [v.x, v.y, v.z]).collect();
        b.push(&r.problem.delassus, &r.problem.free_velocity, &rows, &seed);
    }
    b
}

fn check(name: &str, recs: &[Recorded]) {
    assert!(!recs.is_empty(), "{name}: no contact problems recorded");
    let cfg = cfg_of(&ContactSolverConfig::simulation());
    let mut b = batch_of(recs);
    let stats = solve_host(&cfg, &mut b);
    let mut newton_cases = 0;
    for (p, r) in recs.iter().enumerate() {
        let got = b.impulses(p);
        for (c, (g, w)) in got.iter().zip(&r.impulses).enumerate() {
            for (k, (x, y)) in g.iter().zip([w.x, w.y, w.z]).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "{name} problem {p} (n={}) contact {c}[{k}]: {x:e} vs {y:e}",
                    r.problem.n
                );
            }
        }
        assert_eq!(
            stats[p].0 as usize, r.iterations,
            "{name} problem {p}: iterations"
        );
        assert_eq!(
            stats[p].1.to_bits(),
            r.residual.to_bits(),
            "{name} problem {p}: residual"
        );
        if r.iterations > 16 {
            newton_cases += 1;
        }
    }
    let nmax = recs.iter().map(|r| r.problem.n).max().unwrap();
    eprintln!(
        "{name}: {} problems, n<= {nmax}, {newton_cases} reached the Newton stage — bit-identical",
        recs.len()
    );
}

#[test]
fn dropped_box_is_bit_identical() {
    let m = dropped_box();
    let mut s = m.default_state();
    s.q[0] = 0.4; // tilted, spinning, translating: impact, bounce, slide, rest
    s.q[1] = -0.25;
    s.q[5] = 0.3;
    s.v[2] = 3.0;
    s.v[3] = 0.6;
    check("dropped box", &record(&m, s, 1500, 0.8));
}

#[test]
fn sliding_box_is_bit_identical() {
    let m = dropped_box();
    let mut s = m.default_state();
    s.q[5] = 0.052;
    s.v[3] = 2.5;
    s.v[4] = -0.7;
    check("sliding box", &record(&m, s, 800, 0.3));
}

#[test]
fn foot_chain_is_bit_identical() {
    let m = foot_chain();
    let mut s = m.default_state();
    s.q[5] = 0.42;
    s.q[1] = 0.15;
    s.q[6] = -0.2;
    s.v[3] = 0.4;
    check("foot chain", &record(&m, s, 1500, 1.0));
}

/// Replays a `CVX1` dump (ipse `k1_contact_dump`) when `PHYZ_CVX_DUMP` names
/// one: the K1 stance problems the walk env actually solves. Skips otherwise.
#[test]
fn recorded_dump_is_bit_identical() {
    let Ok(path) = std::env::var("PHYZ_CVX_DUMP") else {
        eprintln!("PHYZ_CVX_DUMP unset; skipping");
        return;
    };
    let d = std::fs::read(&path).expect("read dump");
    assert_eq!(&d[..4], b"CVX1");
    let mut o = 4usize;
    let u = |o: &mut usize| {
        let v = u32::from_le_bytes(d[*o..*o + 4].try_into().unwrap());
        *o += 4;
        v
    };
    let f = |o: &mut usize| {
        let v = f64::from_le_bytes(d[*o..*o + 8].try_into().unwrap());
        *o += 8;
        v
    };
    let count = u(&mut o) as usize;
    let mut recs = Vec::with_capacity(count);
    let mut cfg = None;
    for _ in 0..count {
        let n = u(&mut o) as usize;
        let (reg, tol, rel) = (f(&mut o), f(&mut o), f(&mut o));
        let maxit = u(&mut o);
        let fl = [d[o], d[o + 1], d[o + 2], d[o + 3]];
        o += 4;
        cfg = Some(ConvexCfg {
            regularization: reg,
            tolerance: tol,
            relaxation: rel,
            max_iterations: maxit,
            newton: fl[0] != 0,
            mujoco_compat: fl[1] != 0,
            block_diagonal: fl[2] != 0,
            no_stall_exit: fl[3] != 0,
        });
        let mut v = |k: usize, o: &mut usize| (0..k).map(|_| f(o)).collect::<Vec<f64>>();
        let a = v(9 * n * n, &mut o);
        let b = v(3 * n, &mut o);
        let r = v(3 * n, &mut o);
        let s = v(3 * n, &mut o);
        let w = v(3 * n, &mut o);
        let it = u(&mut o);
        let res = f(&mut o);
        recs.push((n, a, b, r, s, w, it, res));
    }
    let cfg = cfg.unwrap();
    let nmax = recs.iter().map(|r| r.0).max().unwrap();
    let mut bt = ConvexBatch::new(nmax);
    for (_, a, b, r, s, ..) in &recs {
        let rows: Vec<ConvexRow> = r
            .chunks(3)
            .map(|c| ConvexRow {
                mu: c[0],
                bias: c[1],
                impedance: c[2],
            })
            .collect();
        let seed: Vec<[f64; 3]> = s.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
        bt.push(a, b, &rows, &seed);
    }
    let st = solve_host(&cfg, &mut bt);
    for (p, (_, _, _, _, _, w, it, res)) in recs.iter().enumerate() {
        let got: Vec<f64> = bt.impulses(p).into_iter().flatten().collect();
        for (k, (g, x)) in got.iter().zip(w).enumerate() {
            assert_eq!(
                g.to_bits(),
                x.to_bits(),
                "problem {p} component {k}: {g:e} vs {x:e}"
            );
        }
        assert_eq!(st[p].0, *it, "problem {p}: iterations");
        assert_eq!(st[p].1.to_bits(), res.to_bits(), "problem {p}: residual");
    }
    eprintln!("{path}: {count} problems, n <= {nmax} — bit-identical");
}
