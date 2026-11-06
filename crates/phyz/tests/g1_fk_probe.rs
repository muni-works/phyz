//! Regression for the phyz-mjcf body-`quat` transpose: a body's orientation
//! must enter `parent_to_body` as `R_{child<-parent}` (the phyz-urdf
//! convention). The G1's hip-roll body is pitched -10 deg and its knee +10
//! deg, so at zero pose MuJoCo puts the ankle directly under the hip-pitch
//! joint. With the bug the leg slanted back 105 mm and the robot toppled.

use phyz::Simulator;
use phyz_contact::ContactMaterial;
use phyz_mjcf::MjcfLoader;

#[test]
fn g1_zero_pose_ankle_is_under_the_hip() {
    let path = format!("{}/../../models/unitree_g1/g1_23dof.xml", env!("CARGO_MANIFEST_DIR"));
    let model = MjcfLoader::from_file(&path).unwrap().build_model();
    let mut s = model.default_state();
    s.q[5] = 1.0; // free q = [wx,wy,wz,x,y,z]: float clear of the ground
    Simulator::new().step_with_contacts(&model, &mut s, 0.0, &ContactMaterial::default());
    let x = |n: &str| s.body_xform[model.bodies.iter().position(|b| b.name == n).unwrap()].pos.x;
    // MuJoCo FK: hip_yaw at x = +0.0462, knee and ankle at x = 0.
    assert!((x("left_hip_yaw_link") - 0.0462).abs() < 1e-3, "{}", x("left_hip_yaw_link"));
    assert!(x("left_ankle_roll_link").abs() < 1e-3, "{}", x("left_ankle_roll_link"));
}
