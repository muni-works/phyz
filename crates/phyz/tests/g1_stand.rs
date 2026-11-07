//! G1-S0 (ipse `docs/prereg/2026-11-g1-sim.md`): the vendored Unitree G1
//! (`models/unitree_g1/g1_23dof.xml`) stands 5 s on flat ground as a stiff
//! PD statue at Unitree's default pose — a structural check of model,
//! inertials and foot contact, not a controller.
//!
//! Measured, and why the gate is at 10x: at Unitree's published
//! unitree_rl_gym gains (ankle kp 40) pure PD topples forward in ~1 s, and so
//! does 2x/3x/5x. That is physics, not a bug: the ankles' 2 x 40 N·m/rad is
//! far under the gravity stiffness m·g·h ≈ 33 kg x 9.81 x 0.7 ≈ 230 N·m/rad.
//! At published gains the G1 needs a balance policy, as every G1 walker has.
//! `G1_GAIN` overrides the multiplier to reproduce.
//!
//! Pass: pelvis height never drops more than 0.10 m below its settled height,
//! pelvis tilt stays under 0.2 rad, no NaN.

use phyz::Simulator;
use phyz_contact::ContactMaterial;
use phyz_mjcf::MjcfLoader;

/// (name suffix, default q, kp, kd, torque limit) — unitree_rl_gym G1 table.
const PD: [(&str, f64, f64, f64, f64); 11] = [
    ("hip_pitch_joint", -0.1, 100.0, 2.0, 88.0),
    ("hip_roll_joint", 0.0, 100.0, 2.0, 139.0),
    ("hip_yaw_joint", 0.0, 100.0, 2.0, 88.0),
    ("knee_joint", 0.3, 150.0, 4.0, 139.0),
    ("ankle_pitch_joint", -0.2, 40.0, 2.0, 50.0),
    ("ankle_roll_joint", 0.0, 40.0, 2.0, 50.0),
    ("waist_yaw_joint", 0.0, 100.0, 2.0, 88.0),
    ("shoulder_pitch_joint", 0.0, 40.0, 1.0, 25.0),
    ("shoulder_roll_joint", 0.0, 40.0, 1.0, 25.0),
    ("shoulder_yaw_joint", 0.0, 40.0, 1.0, 25.0),
    ("elbow_joint", 0.0, 40.0, 1.0, 25.0),
];

fn gains(name: &str) -> (f64, f64, f64, f64) {
    PD.iter()
        .find(|p| name.ends_with(p.0))
        .map(|p| (p.1, p.2, p.3, p.4))
        .unwrap_or((0.0, 40.0, 1.0, 25.0)) // wrist roll
}

#[test]
fn g1_stands_five_seconds_as_a_stiff_pd_statue() {
    let path = format!("{}/../../models/unitree_g1/g1_23dof.xml", env!("CARGO_MANIFEST_DIR"));
    let mut model = MjcfLoader::from_file(&path).unwrap().build_model();
    model.dt = 0.002;
    let mut state = model.default_state();
    let hinges: Vec<(usize, usize, (f64, f64, f64, f64))> = model
        .joints
        .iter()
        .enumerate()
        
        .filter(|(_, j)| j.name != "floating_base_joint")
        .map(|(i, j)| (model.q_offsets[i], model.v_offsets[i], gains(&j.name)))
        .collect();
    assert_eq!(hinges.len(), 23);
    for &(qi, _, (q0, ..)) in &hinges {
        state.q[qi] = q0;
    }
    // Free q = [wx,wy,wz,x,y,z]; lower the base so knees-bent soles touch.
    state.q[5] -= 0.04;
    let gain: f64 = std::env::var("G1_GAIN").ok().and_then(|s| s.parse().ok()).unwrap_or(10.0);
    let sim = Simulator::new();
    let mat = ContactMaterial { stiffness: 1.0e5, damping: 2.0e3, friction: 0.8, ..Default::default() };
    let steps = (5.0 / model.dt) as usize;
    let mut settled = None;
    let (mut min_z, mut max_tilt) = (f64::INFINITY, 0.0f64);
    for k in 0..steps {
                for (a, &(qi, vi, (q0, kp, kd, lim))) in hinges.iter().enumerate() {
            let tau = gain * kp * (q0 - state.q[qi]) - gain.sqrt() * kd * state.v[vi];
            state.ctrl[a] = tau.clamp(-lim, lim);
        }
        sim.step_with_contacts(&model, &mut state, 0.0, &mat);
        let z = state.q[5];
        assert!(z.is_finite(), "NaN at step {k}");
        if k == (0.5 / model.dt) as usize {
            settled = Some(z);
        }
        if settled.is_some() {
            min_z = min_z.min(z);
            max_tilt = max_tilt.max(state.q[0].abs().max(state.q[1].abs()));
        }
    }
    let s = settled.unwrap();
    eprintln!("G1-S0: settled z {s:.3}, min z {min_z:.3}, max tilt {max_tilt:.3} rad");
    assert!(s - min_z < 0.10, "sagged {:.3} m", s - min_z);
    assert!(max_tilt < 0.2, "tilted {max_tilt:.3} rad");
}
