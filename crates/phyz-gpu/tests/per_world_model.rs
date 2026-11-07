//! Per-world model data — body tables (mass, COM, inertia, armature),
//! contact friction and heightfield terrain — is the single-world
//! simulator, bit for bit.
//!
//! World `w` of a batch given `set_world_models` / `set_world_friction` /
//! `set_world_heightfields` must step exactly like a one-world simulator
//! built from world `w`'s own model, friction and terrain. Same kernels,
//! same expressions, a different row of the same tables: not a tolerance.
#![cfg(feature = "cuda-host")]

use phyz_gpu::cuda::{HostBatchSimulator, StepMode};
use phyz_gpu::{BodyContactGains, PdDof};
use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Heightfield, Joint, JointType, Model, ModelBuilder, State};

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

/// Free base + two hinged links with box geometry, PD on the hinges.
fn rig() -> Model {
    let half = Vec3::new(0.12, 0.1, 0.06);
    let mut model = ModelBuilder::new()
        .gravity(Vec3::new(0.0, 0.0, -GRAVITY))
        .dt(0.001)
        .add_body("base", -1, Joint::free(SpatialTransform::identity()), box_inertia(3.0, half))
        .add_body("link", 0, hinge(-0.2), box_inertia(1.0, half))
        .add_body("foot", 1, hinge(-0.2), box_inertia(0.5, half))
        .build();
    for b in &mut model.bodies {
        b.geometry = Some(phyz_model::Geometry::Box { half_extents: half });
    }
    model
}

/// The DR a walk env draws per world: trunk mass and COM, joint armature.
fn variant(w: usize) -> Model {
    let mut m = rig();
    let f = w as f64;
    let t = &mut m.bodies[0].inertia;
    let s = 1.0 + 0.15 * f;
    let com = t.com + Vec3::new(0.01 * f, -0.007 * f, 0.004 * f);
    *t = SpatialInertia::new(t.mass * s, com, t.inertia * s);
    let ji = m.bodies[1].joint_idx;
    m.joints[ji].armature = 0.01 * f;
    let ji = m.bodies[2].joint_idx;
    m.joints[ji].damping = 0.05 * f;
    m
}

fn spawn(model: &Model) -> State {
    let mut s = model.default_state();
    s.q[5] = 0.55;
    s.q[1] = 0.2;
    s.q[6] = 0.3;
    s.q[7] = -0.4;
    s.v[3] = 0.4;
    s.v[1] = 1.0;
    s
}

fn pd(model: &Model) -> Vec<PdDof> {
    (6..model.nv)
        .map(|i| PdDof { q_index: i, v_index: i, kp: 30.0, kd: 1.5, max_force: 40.0 })
        .collect()
}

fn terrain(w: usize) -> Heightfield {
    let mut h = Heightfield::new(Vec3::new(-1.0, -1.0, 0.0), 0.25, 9, 9);
    for (i, z) in h.heights.iter_mut().enumerate() {
        *z = 0.01 * ((i * (w + 3)) % 7) as f32;
    }
    h
}

const MU: [f64; 4] = [0.9, 0.3, 0.6, 1.2];
const STEPS: usize = 300;

/// One world, its own model / friction / terrain.
fn single(mode: StepMode, w: usize, terrain_on: bool) -> State {
    let m = variant(w);
    let mut sim = HostBatchSimulator::new(m.clone(), 1).unwrap();
    sim.set_step_mode(mode);
    // Contact gains are per body, from the BASE model on both sides (the
    // batch has one gain table; DR here is mass, not contact stiffness).
    let g = BodyContactGains::uniform_frequency(&rig(), 250.0, 1.0);
    let hf = terrain_on.then(|| terrain(w));
    sim.enable_contact_impulse(0.0, MU[w], &g, &[], hf.as_ref()).unwrap();
    sim.enable_pd_control(&pd(&m)).unwrap();
    sim.load_states(&[spawn(&m)]);
    sim.set_position_targets(&[vec![0.1, -0.2]]).unwrap();
    sim.step_many(STEPS).unwrap();
    sim.readback_states().remove(0)
}

/// Four worlds on one shared simulator with per-world data.
fn batched(mode: StepMode, terrain_on: bool) -> Vec<State> {
    let n = MU.len();
    let base = rig();
    let mut sim = HostBatchSimulator::new(base.clone(), n).unwrap();
    sim.set_step_mode(mode);
    let g = BodyContactGains::uniform_frequency(&base, 250.0, 1.0);
    let hf0 = terrain_on.then(|| terrain(0));
    sim.enable_contact_impulse(0.0, 0.5, &g, &[], hf0.as_ref()).unwrap();
    sim.enable_pd_control(&pd(&base)).unwrap();
    let models: Vec<Model> = (0..n).map(variant).collect();
    sim.set_world_models(&models).unwrap();
    sim.set_world_friction(&MU).unwrap();
    if terrain_on {
        let fields: Vec<Heightfield> = (0..n).map(terrain).collect();
        sim.set_world_heightfields(&fields).unwrap();
    }
    let states: Vec<State> = models.iter().map(spawn).collect();
    sim.load_states(&states);
    sim.set_position_targets(&vec![vec![0.1, -0.2]; n]).unwrap();
    sim.step_many(STEPS).unwrap();
    sim.readback_states()
}

fn assert_bits(a: &State, b: &State, w: usize) {
    for (i, (x, y)) in a.q.iter().zip(b.q.iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "world {w} q[{i}]: {x} vs {y}");
    }
    for (i, (x, y)) in a.v.iter().zip(b.v.iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "world {w} v[{i}]: {x} vs {y}");
    }
}

fn check(mode: StepMode, terrain_on: bool) {
    let b = batched(mode, terrain_on);
    // The worlds must actually differ, or the test proves nothing.
    assert_ne!(b[0].q[5].to_bits(), b[3].q[5].to_bits(), "per-world data had no effect");
    for w in 0..MU.len() {
        assert_bits(&b[w], &single(mode, w, terrain_on), w);
    }
}

#[test]
fn per_world_models_and_friction_fused() {
    check(StepMode::Fused, false);
}

#[test]
fn per_world_models_and_friction_unfused() {
    check(StepMode::Unfused, false);
}

#[test]
fn per_world_terrain_fused() {
    check(StepMode::Fused, true);
}

#[test]
fn per_world_terrain_fission() {
    check(StepMode::Fission, true);
}

#[test]
fn set_model_returns_to_shared() {
    let n = 2;
    let base = rig();
    let mut sim = HostBatchSimulator::new(base.clone(), n).unwrap();
    let g = BodyContactGains::uniform_frequency(&base, 250.0, 1.0);
    sim.enable_contact_impulse(0.0, 0.9, &g, &[], None).unwrap();
    sim.set_world_models(&[variant(1), variant(2)]).unwrap();
    sim.set_model(base.clone()).unwrap();
    sim.load_states(&[spawn(&base), spawn(&base)]);
    sim.step_many(100).unwrap();
    let s = sim.readback_states();
    assert_bits(&s[0], &s[1], 1);
}

#[test]
fn structure_mismatch_is_refused() {
    let base = rig();
    let mut sim = HostBatchSimulator::new(base.clone(), 2).unwrap();
    let mut bad = rig();
    let ji = bad.bodies[1].joint_idx;
    bad.joints[ji].joint_type = JointType::Prismatic;
    assert!(sim.set_world_models(&[base.clone(), bad]).is_err());
    assert!(sim.set_world_models(&[base]).is_err(), "wrong count");
}

/// `update_world_models` / `update_world_friction` in place, after a
/// different first table, are the batch built with the final table: what a
/// per-world DR reset in an RL env does every control step.
#[test]
fn in_place_updates_match_a_fresh_table() {
    let n = MU.len();
    let base = rig();
    let mut sim = HostBatchSimulator::new(base.clone(), n).unwrap();
    let g = BodyContactGains::uniform_frequency(&base, 250.0, 1.0);
    sim.enable_contact_impulse(0.0, 0.5, &g, &[], None).unwrap();
    sim.enable_pd_control(&pd(&base)).unwrap();
    let addr = sim.state_addrs();
    // Start from the wrong table, then fix worlds 1..n in place.
    let models: Vec<Model> = (0..n).map(variant).collect();
    sim.set_world_models(&vec![variant(0); n]).unwrap();
    sim.set_world_friction(&[0.5; 4]).unwrap();
    let upd: Vec<(usize, &Model)> = (1..n).map(|w| (w, &models[w])).collect();
    sim.update_world_models(&upd).unwrap();
    let fr: Vec<(usize, f64)> = (0..n).map(|w| (w, MU[w])).collect();
    sim.update_world_friction(&fr).unwrap();
    assert_eq!(sim.state_addrs(), addr, "state buffers moved");
    let states: Vec<State> = models.iter().map(spawn).collect();
    sim.load_states(&states);
    sim.set_position_targets(&vec![vec![0.1, -0.2]; n]).unwrap();
    sim.step_many(STEPS).unwrap();
    let got = sim.readback_states();
    let want = batched(StepMode::Fused, false);
    for w in 0..n {
        assert_bits(&got[w], &want[w], w);
    }
    assert!(sim.update_world_models(&[(n, &models[0])]).is_err());
}
