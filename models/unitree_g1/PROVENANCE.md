# Unitree G1 (23-DOF, rev 1.0) — provenance

**Source:** `unitreerobotics/unitree_ros`, `robots/g1_description/g1_23dof_rev_1_0.xml`,
commit `daadf41ee9afce8f90fdc09a98506012691fa122`.
**Licence:** BSD 3-Clause, © 2016-2022 HangZhou YuShu Technology (Unitree Robotics).
Full text in [`LICENSE.unitree`](LICENSE.unitree). Redistribution of the modified file is
permitted with the notice retained (it is, in the file header and here).

`mode_machine` 4 per Unitree's README: hip pitch/roll gear {14.3, 22.5}, 1 waist DOF,
5-DOF arms, no hands. 23 actuated joints + free base = nv 29.

## Kept verbatim
Kinematic tree, body masses/inertials, joint axes and ranges, `actuatorfrcrange`
torque limits (hip 88/139, knee 139, ankle 50, waist 88, arm 25 N·m), 4 foot
contact spheres per sole (r = 5 mm), primitive cylinders, sites, IMU sensors.

## Changed
| change | why |
|---|---|
| all `type="mesh"` geoms and the `<asset>` mesh list removed | STL meshes are not vendored; foot spheres + primitives carry contact |
| scene (floor, skybox, lights) removed | phyz callers add their own ground |
| `armature` added per joint | the URDF/MJCF has none and armature is load-bearing for PD stability (ipse `dojo-morphology` finding). Values are reflected rotor inertia per actuator class, as used by mjlab's G1 config (Apache-2.0): 7520-14 (hip pitch/yaw, waist) 0.010177520; 7520-22 (hip roll, knee) 0.025101925; ankle = 2 × 5020 in parallel 0.007219450; 5020 (arms) 0.003609725 kg·m² |
| foot geoms given explicit `type="sphere"`, `friction="0.6"` | MuJoCo's implicit default type is sphere; explicit for phyz; 0.6 matches menagerie's foot class |

Velocity limits are not in Unitree's file; they are not invented here.
