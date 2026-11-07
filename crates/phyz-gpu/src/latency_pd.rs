//! **Latency PD actuator** for the CUDA batch simulator.
//!
//! [`crate::pd_pipeline`]'s servo reads the state of the step it runs in and
//! a target the host wrote just before. A legged robot's motor board does
//! neither: the command it executes was published some milliseconds ago,
//! and the joint state its PD closes on is itself a few steps stale. Before
//! this pass, an RL plant that modelled that had to launch its own actuator
//! kernel before every physics step, which split a control period into one
//! `step_many(1)` per substep and kept the fused kernel and graph capture
//! from ever covering more than one step.
//!
//! This pass runs inside the step (once per physics step, one thread per
//! world, in the fused kernel and in the per-pass sequence alike), so one
//! [`BatchSim::step_many`](crate::cuda::BatchSim::step_many)`(sub)` covers
//! the whole control period. Per world and step `G`:
//!
//! - the state at the start of the step goes into a q/v history ring;
//! - the PD target is the one published for the control period containing
//!   `G − act_lat`, or the spec's default pose while `G − act_lat < G0`
//!   (`G0` = the world's reset step, [`BatchSim::latency_reset`]);
//! - the PD reads the state at `max(G − pd_lag, G0)` from the ring;
//! - `τ = kp (target − q) − kd q̇`, Coulomb joint friction `min(jf, |τ|)`
//!   subtracted, clamp to ± effort — in f64, written to `ctrl` with every
//!   other DOF zeroed;
//! - the world's base push (torque on `v[0..3]`, force on `v[3..6]`) added;
//! - an optional gantry pin holds the free base (`q[0..6]` set, `v[0..6]` 0);
//! - `τ / sub` accumulates into the period's mean torque.
//!
//! [`BatchSim::latency_readout`] returns, per world, `q`, `v`, the mean
//! torque, the delayed state at `G − min(G − G0, obs_lat)` and the state at
//! the start of the period's last step — one launch and one copy.
//!
//! [`BatchSim::latency_reset`]: crate::cuda::BatchSim::latency_reset
//! [`BatchSim::latency_readout`]: crate::cuda::BatchSim::latency_readout

/// Shape of the actuator, fixed for the simulator's life.
#[derive(Clone, Debug, PartialEq)]
pub struct LatencyPdSpec {
    /// Per actuated joint: its index into `q`.
    pub q_idx: Vec<usize>,
    /// Per actuated joint: its index into `v` (and `ctrl`).
    pub v_idx: Vec<usize>,
    /// Per actuated joint: the target before a world's first command.
    pub default: Vec<f32>,
    /// Per actuated joint: the effort clamp.
    pub effort: Vec<f32>,
    /// Physics steps per control period.
    pub sub: usize,
    /// State history ring depth, in steps. Must exceed the largest PD lag
    /// and observation latency.
    pub ring: usize,
    /// Published-target ring depth, in control periods. Must exceed the
    /// largest actuation latency in periods, plus one.
    pub target_ring: usize,
}

impl LatencyPdSpec {
    /// Actuated joints.
    pub fn nj(&self) -> usize {
        self.q_idx.len()
    }

    /// Floats per world in the `par` row [`BatchSim::set_latency_params`]
    /// takes: `kp[nj] kd[nj] jfric[nj] push_f[3] push_t[3] pin_flag pin_q[6]`.
    ///
    /// [`BatchSim::set_latency_params`]: crate::cuda::BatchSim::set_latency_params
    pub fn par_stride(&self) -> usize {
        3 * self.nj() + 13
    }

    /// Floats per world in [`BatchSim::latency_readout`]'s result.
    ///
    /// [`BatchSim::latency_readout`]: crate::cuda::BatchSim::latency_readout
    pub fn readout_stride(&self, nq: usize, nv: usize) -> usize {
        3 * (nq + nv) + self.nj()
    }

    pub(crate) fn check(&self, nq: usize, nv: usize) -> Result<(), String> {
        let n = self.nj();
        if n == 0 || self.v_idx.len() != n || self.default.len() != n || self.effort.len() != n {
            return Err(
                "LatencyPdSpec: q_idx/v_idx/default/effort must be equal, non-empty".into(),
            );
        }
        if self.q_idx.iter().any(|&i| i >= nq) || self.v_idx.iter().any(|&i| i >= nv) {
            return Err("LatencyPdSpec: joint index out of range".into());
        }
        if nv < 6 {
            // Pushes and the pin address v[0..6]; a fixed-base model has none.
            return Err("LatencyPdSpec: needs a free base (nv >= 6)".into());
        }
        if self.sub == 0 || self.ring < 2 || self.target_ring < 2 {
            return Err("LatencyPdSpec: sub >= 1, ring >= 2, target_ring >= 2".into());
        }
        Ok(())
    }

    /// The packed table the kernel reads (see `lat_pd_thread`).
    pub(crate) fn pack_table(&self) -> Vec<f32> {
        let u = |x: usize| f32::from_bits(x as u32);
        let mut t = vec![u(self.ring), u(self.target_ring), u(self.sub), u(self.nj())];
        for k in 0..self.nj() {
            t.extend([
                u(self.q_idx[k]),
                u(self.v_idx[k]),
                self.default[k],
                self.effort[k],
            ]);
        }
        t
    }
}

/// Per-world latencies, in physics steps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorldLatency {
    /// Command latency: the target in force at step `G` is the one
    /// published for the period containing `G − act`.
    pub act: u32,
    /// Motor-board PD lag: the PD closes on the state at `G − pd_lag`.
    pub pd_lag: u32,
    /// Observation latency read back by `latency_readout`.
    pub obs: u32,
}
