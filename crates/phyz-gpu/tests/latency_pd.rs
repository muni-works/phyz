//! The latency PD actuator ([`phyz_gpu::latency_pd`]): a whole control
//! period in one `step_many(sub)` is the per-step sequence, bit for bit, in
//! every step mode; and the readout's rings hold the states they claim to.
#![cfg(feature = "cuda-host")]

use phyz_gpu::cuda::{HostBatchSimulator, StepMode};
use phyz_gpu::{BodyContactGains, LatencyPdSpec, WorldLatency};
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Joint, JointType, Model, ModelBuilder, State};

const SUB: usize = 5;

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

fn hinge(offset_z: f64) -> Joint {
    Joint {
        joint_type: JointType::Revolute,
        parent_to_joint: SpatialTransform::new(Mat3::identity(), Vec3::new(0.0, 0.0, offset_z)),
        axis: Vec3::new(0.0, 1.0, 0.0),
        ..Joint::default()
    }
}

/// Free base, two hinges, box geometry: lands on the ground in the run.
fn rig() -> Model {
    let half = Vec3::new(0.12, 0.1, 0.06);
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(0.001)
        .add_body(
            "base",
            -1,
            Joint::free(SpatialTransform::identity()),
            box_inertia(3.0, half),
        )
        .add_body("link", 0, hinge(-0.2), box_inertia(1.0, half))
        .add_body("foot", 1, hinge(-0.2), box_inertia(0.5, half))
        .build();
    for b in &mut model.bodies {
        b.geometry = Some(phyz_model::Geometry::Box { half_extents: half });
    }
    model
}

const NW: usize = 4;

fn states(model: &Model) -> Vec<State> {
    (0..NW)
        .map(|i| {
            let mut s = model.default_state();
            s.q[5] = 0.5 + 0.05 * i as f64;
            let nq = s.q.len();
            s.q[nq - 2] = 0.3;
            s.q[nq - 1] = -0.4;
            s.v[3] = 0.4 - 0.2 * i as f64;
            s
        })
        .collect()
}

fn spec(model: &Model) -> LatencyPdSpec {
    LatencyPdSpec {
        q_idx: vec![model.nq - 2, model.nq - 1],
        v_idx: vec![model.nv - 2, model.nv - 1],
        default: vec![0.2, -0.3],
        effort: vec![25.0, 25.0],
        sub: SUB,
        ring: 8,
        target_ring: 4,
    }
}

fn lats() -> Vec<WorldLatency> {
    // World 0 has no latency at all; the others mix act/pd/obs lags.
    vec![
        WorldLatency {
            act: 0,
            pd_lag: 0,
            obs: 0,
        },
        WorldLatency {
            act: 3,
            pd_lag: 1,
            obs: 2,
        },
        WorldLatency {
            act: 7,
            pd_lag: 2,
            obs: 4,
        },
        WorldLatency {
            act: 12,
            pd_lag: 0,
            obs: 6,
        },
    ]
}

fn par(sp: &LatencyPdSpec, period: usize) -> Vec<f32> {
    let nj = sp.nj();
    let mut p = vec![0f32; NW * sp.par_stride()];
    for w in 0..NW {
        let r = &mut p[w * sp.par_stride()..(w + 1) * sp.par_stride()];
        for k in 0..nj {
            r[k] = 40.0 + 5.0 * w as f32;
            r[nj + k] = 1.5;
            r[2 * nj + k] = 0.3 * k as f32;
        }
        if w == 2 && period % 3 == 1 {
            r[3 * nj] = 15.0; // push force x
            r[3 * nj + 5] = 2.0; // push torque z
        }
        if w == 3 && period < 2 {
            r[3 * nj + 6] = 1.0; // gantry pin, at the initial base pose
            let s0 = &states(&rig())[3];
            for i in 0..6 {
                r[3 * nj + 7 + i] = s0.q[i] as f32;
            }
        }
    }
    p
}

fn targets(period: usize) -> Vec<f32> {
    (0..NW * 2)
        .map(|i| 0.4 * ((period * 7 + i) as f32 * 0.37).sin())
        .collect()
}

/// `periods` control periods; world 1 resets at period 6. `per_step` issues
/// every step on its own, otherwise one `step_many(SUB)` per period.
fn run(mode: StepMode, per_step: bool, periods: usize) -> (Vec<State>, Vec<Vec<f32>>) {
    let model = rig();
    let mut sim = HostBatchSimulator::new(model.clone(), NW).unwrap();
    sim.set_step_mode(mode);
    let g = BodyContactGains::uniform_frequency(&model, 60.0, 1.0);
    sim.enable_contact_impulse(0.0, 0.7, &g, &[], None).unwrap();
    let sp = spec(&model);
    sim.enable_latency_pd(sp.clone()).unwrap();
    let init = states(&model);
    sim.load_states(&init);
    let mut outs = Vec::new();
    for p in 0..periods {
        if p == 6 {
            let mut st = sim.readback_states();
            st[1] = init[1].clone();
            sim.load_states(&st);
            sim.latency_reset(&[1]).unwrap();
        }
        sim.set_latency_params(&par(&sp, p), &lats()).unwrap();
        sim.publish_latency_targets(&targets(p)).unwrap();
        if per_step {
            for _ in 0..SUB {
                sim.step_many(1).unwrap();
            }
        } else {
            sim.step_many(SUB).unwrap();
        }
        outs.push(sim.latency_readout().unwrap());
    }
    (sim.readback_states(), outs)
}

#[test]
fn period_in_one_call_is_bit_identical_in_every_mode() {
    let (ref_states, ref_outs) = run(StepMode::Unfused, true, 14);
    for (label, mode, per_step) in [
        ("unfused(sub)", StepMode::Unfused, false),
        ("fused(1)", StepMode::Fused, true),
        ("fused(sub)", StepMode::Fused, false),
        ("fission(sub)", StepMode::Fission, false),
    ] {
        let (st, outs) = run(mode, per_step, 14);
        for (w, (a, b)) in ref_states.iter().zip(&st).enumerate() {
            for j in 0..a.q.len() {
                assert_eq!(
                    a.q[j].to_bits(),
                    b.q[j].to_bits(),
                    "{label} world {w} q[{j}]"
                );
            }
            for j in 0..a.v.len() {
                assert_eq!(
                    a.v[j].to_bits(),
                    b.v[j].to_bits(),
                    "{label} world {w} v[{j}]"
                );
            }
        }
        for (p, (a, b)) in ref_outs.iter().zip(&outs).enumerate() {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                assert_eq!(x.to_bits(), y.to_bits(), "{label} period {p} readout[{i}]");
            }
        }
    }
}

/// The readout's delayed and last-step rows are the states the simulator
/// held at those steps, and the actuator really moves the joints.
#[test]
fn readout_rings_hold_the_right_states() {
    let model = rig();
    let (nq, nv) = (model.nq, model.nv);
    let mut sim = HostBatchSimulator::new(model.clone(), NW).unwrap();
    let g = BodyContactGains::uniform_frequency(&model, 60.0, 1.0);
    sim.enable_contact_impulse(0.0, 0.7, &g, &[], None).unwrap();
    let sp = spec(&model);
    sim.enable_latency_pd(sp.clone()).unwrap();
    sim.load_states(&states(&model));
    let stride = sp.readout_stride(nq, nv);
    let mut hist: Vec<Vec<State>> = vec![sim.readback_states()];
    for p in 0..6 {
        sim.set_latency_params(&par(&sp, p), &lats()).unwrap();
        sim.publish_latency_targets(&targets(p)).unwrap();
        for _ in 0..SUB {
            sim.step_many(1).unwrap();
            hist.push(sim.readback_states());
        }
        let out = sim.latency_readout().unwrap();
        let gnow = hist.len() - 1;
        for (w, l) in lats().iter().enumerate() {
            let o = &out[w * stride..(w + 1) * stride];
            let cur = &hist[gnow][w];
            for j in 0..nq {
                assert_eq!(o[j], cur.q[j] as f32, "period {p} world {w} q[{j}]");
            }
            // The pin rewrites the base before the ring sees it, so a pinned
            // world's ring is not the previous readback; skip it while pinned.
            if w == 3 && p < 3 {
                continue;
            }
            let last = &hist[gnow - 1][w];
            let ol = &o[nq + nv + sp.nj() + nq + nv..];
            for j in 0..nq {
                assert_eq!(ol[j], last.q[j] as f32, "period {p} world {w} last q[{j}]");
            }
            let m = (l.obs as usize).min(gnow);
            assert_eq!(sim.latency_has_delayed(w), m > 0);
            if m > 0 {
                let d = &hist[gnow - m][w];
                let od = &o[nq + nv + sp.nj()..];
                for j in 0..nv {
                    assert_eq!(
                        od[nq + j],
                        d.v[j] as f32,
                        "period {p} world {w} delayed v[{j}]"
                    );
                }
            }
        }
    }
    let end = sim.readback_states();
    assert!(
        (end[0].q[nq - 1] - states(&model)[0].q[nq - 1]).abs() > 1e-3,
        "the actuator never moved the joint"
    );
}

#[test]
fn publish_mid_period_is_refused() {
    let model = rig();
    let mut sim = HostBatchSimulator::new(model.clone(), NW).unwrap();
    sim.enable_latency_pd(spec(&model)).unwrap();
    sim.load_states(&states(&model));
    sim.step_many(2).unwrap();
    assert!(sim.publish_latency_targets(&targets(0)).is_err());
}
