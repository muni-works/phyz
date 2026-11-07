//! `phyz_contact`'s convex contact solve on the device, f64, one thread per
//! problem — the kernel is `cuda/phyz_convex.cu`.
//!
//! This is the solver core alone: it takes assembled problems (Delassus `A`,
//! free velocity `b`, per-contact `mu` / `bias` / effective impedance and a
//! warm-start seed) and returns impulses. Assembly and contact detection are
//! separate work. With FMA contraction off on both sides the result is meant
//! to be the CPU solver's to the bit; `tests/convex_parity.rs` pins that.

/// The kernel source, compiled by NVRTC (device) or `cc` (`cuda-host`).
pub const CONVEX_KERNEL_SOURCE: &str = include_str!("../../cuda/phyz_convex.cu");

/// `phyz_contact::ContactSolverConfig`, the fields the forward solve reads.
#[derive(Clone, Copy, Debug)]
pub struct ConvexCfg {
    /// Regularization floor.
    pub regularization: f64,
    /// Sweep-residual tolerance.
    pub tolerance: f64,
    /// Over-relaxation factor.
    pub relaxation: f64,
    /// Iteration budget.
    pub max_iterations: u32,
    /// Active-set Newton stage on.
    pub newton: bool,
    /// MuJoCo-compatible tangential regularization.
    pub mujoco_compat: bool,
    /// `true` = `ContactCoupling::BlockDiagonal`; otherwise `Full`.
    /// (`PerBody` is not supported here.)
    pub block_diagonal: bool,
    /// Spend the whole budget (no stagnation exit).
    pub no_stall_exit: bool,
}

/// One contact row: `mu`, `bias`, `effective_impedance()`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ConvexRow {
    /// Friction coefficient.
    pub mu: f64,
    /// Normal bias velocity (impact-scaled).
    pub bias: f64,
    /// `ContactRow::effective_impedance()`.
    pub impedance: f64,
}

const ROW_STRIDE: usize = 3;

/// A packed batch of problems at a common contact capacity `nmax`.
#[derive(Clone, Debug, Default)]
pub struct ConvexBatch {
    /// Contact capacity per problem.
    pub nmax: usize,
    /// Contact count per problem.
    pub ncont: Vec<u32>,
    /// Delassus blocks, `[p][3nmax][3nmax]`.
    pub a: Vec<f64>,
    /// Free velocities, `[p][3nmax]`.
    pub b: Vec<f64>,
    /// Rows, `[p][nmax][3]`.
    pub rows: Vec<f64>,
    /// Warm-start seeds in; impulses out (`[normal, u, w]` per contact).
    pub f: Vec<f64>,
}

impl ConvexBatch {
    /// An empty batch at capacity `nmax` (or a device solver on `ordinal`).
    pub fn new(nmax: usize) -> Self {
        Self {
            nmax: nmax.max(1),
            ..Default::default()
        }
    }
    /// Number of problems.
    pub fn len(&self) -> usize {
        self.ncont.len()
    }
    /// Whether there are no problems.
    pub fn is_empty(&self) -> bool {
        self.ncont.is_empty()
    }
    /// Append one problem: `delassus` is row-major `3n x 3n`, `free_velocity`
    /// `3n`, `seed` up to `n` impulses (missing entries start at zero).
    pub fn push(
        &mut self,
        delassus: &[f64],
        free_velocity: &[f64],
        rows: &[ConvexRow],
        seed: &[[f64; 3]],
    ) {
        let n = rows.len();
        assert!(
            n <= self.nmax,
            "problem has {n} contacts, batch capacity {}",
            self.nmax
        );
        let dim = 3 * n;
        assert_eq!(delassus.len(), dim * dim);
        assert_eq!(free_velocity.len(), dim);
        let d = 3 * self.nmax;
        let mut a = vec![0.0; d * d];
        a[..dim * dim].copy_from_slice(delassus);
        self.a.extend_from_slice(&a);
        let mut b = vec![0.0; d];
        b[..dim].copy_from_slice(free_velocity);
        self.b.extend_from_slice(&b);
        let mut r = vec![0.0; self.nmax * ROW_STRIDE];
        for (c, row) in rows.iter().enumerate() {
            r[c * ROW_STRIDE] = row.mu;
            r[c * ROW_STRIDE + 1] = row.bias;
            r[c * ROW_STRIDE + 2] = row.impedance;
        }
        self.rows.extend_from_slice(&r);
        let mut f = vec![0.0; d];
        for (c, s) in seed.iter().take(n).enumerate() {
            f[3 * c..3 * c + 3].copy_from_slice(s);
        }
        self.f.extend_from_slice(&f);
        self.ncont.push(n as u32);
    }
    /// Impulses of problem `p`.
    pub fn impulses(&self, p: usize) -> Vec<[f64; 3]> {
        let d = 3 * self.nmax;
        (0..self.ncont[p] as usize)
            .map(|c| {
                let o = p * d + 3 * c;
                [self.f[o], self.f[o + 1], self.f[o + 2]]
            })
            .collect()
    }
    fn scratch_len(&self) -> usize {
        let d = 3 * self.nmax;
        (9 * self.nmax + d * d + 3 * d) * self.len().max(1)
    }
}

/// Per-problem `(iterations, residual)`, as `ContactSolution` reports them.
pub type ConvexStats = Vec<(u32, f64)>;

fn stats(raw: &[f64]) -> ConvexStats {
    raw.chunks(2).map(|s| (s[0] as u32, s[1])).collect()
}

#[cfg(feature = "cuda-host")]
unsafe extern "C" {
    fn phyz_host_convex_solve(
        nprob: u32,
        nmax: u32,
        regularization: f64,
        tolerance: f64,
        relaxation: f64,
        max_iterations: u32,
        newton: u32,
        mujoco_compat: u32,
        coupling: u32,
        no_stall_exit: u32,
        ncont: *const u32,
        a: *const f64,
        b: *const f64,
        rows: *const f64,
        f: *mut f64,
        stat: *mut f64,
        scratch: *mut f64,
    );
}

/// Solve every problem in `batch` on the host (the kernel source as C++).
#[cfg(feature = "cuda-host")]
pub fn solve_host(cfg: &ConvexCfg, batch: &mut ConvexBatch) -> ConvexStats {
    let mut stat = vec![0.0; 2 * batch.len().max(1)];
    let mut scratch = vec![0.0; batch.scratch_len()];
    // SAFETY: every buffer is sized for `batch.len()` problems at `nmax`.
    unsafe {
        phyz_host_convex_solve(
            batch.len() as u32,
            batch.nmax as u32,
            cfg.regularization,
            cfg.tolerance,
            cfg.relaxation,
            cfg.max_iterations,
            cfg.newton as u32,
            cfg.mujoco_compat as u32,
            if cfg.block_diagonal { 2 } else { 0 },
            cfg.no_stall_exit as u32,
            batch.ncont.as_ptr(),
            batch.a.as_ptr(),
            batch.b.as_ptr(),
            batch.rows.as_ptr(),
            batch.f.as_mut_ptr(),
            stat.as_mut_ptr(),
            scratch.as_mut_ptr(),
        );
    }
    stats(&stat[..2 * batch.len()])
}

/// The device solver: one NVRTC module, one kernel.
#[cfg(feature = "cuda")]
pub struct CudaConvex {
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    func: cudarc::driver::CudaFunction,
}

#[cfg(feature = "cuda")]
impl CudaConvex {
    /// An empty batch at capacity `nmax` (or a device solver on `ordinal`).
    pub fn new(ordinal: usize) -> Result<Self, String> {
        use cudarc::nvrtc::CompileOptions;
        let e = |w: &'static str| move |x: cudarc::driver::DriverError| format!("{w}: {x}");
        let ctx = cudarc::driver::CudaContext::new(ordinal).map_err(e("CUDA context"))?;
        let opts = CompileOptions {
            use_fast_math: Some(false),
            prec_div: Some(true),
            prec_sqrt: Some(true),
            fmad: Some(false),
            name: Some("phyz_convex.cu".into()),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(CONVEX_KERNEL_SOURCE, opts)
            .map_err(|x| format!("NVRTC compile of phyz_convex.cu: {x}"))?;
        let module = ctx.load_module(ptx).map_err(e("load PTX"))?;
        let func = module
            .load_function("phyz_convex_solve")
            .map_err(e("load kernel"))?;
        Ok(Self {
            stream: ctx.default_stream(),
            func,
        })
    }

    /// Solve `batch` on the device. Returns the stats and the kernel's wall
    /// time in seconds (uploads and downloads excluded).
    pub fn solve(
        &self,
        cfg: &ConvexCfg,
        batch: &mut ConvexBatch,
        block: u32,
    ) -> Result<(ConvexStats, f64), String> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        let e = |w: &'static str| move |x: cudarc::driver::DriverError| format!("{w}: {x}");
        let s = &self.stream;
        let nprob = batch.len() as u32;
        let ncont = s.clone_htod(&batch.ncont).map_err(e("upload ncont"))?;
        let a = s.clone_htod(&batch.a).map_err(e("upload A"))?;
        let b = s.clone_htod(&batch.b).map_err(e("upload b"))?;
        let rows = s.clone_htod(&batch.rows).map_err(e("upload rows"))?;
        let mut f = s.clone_htod(&batch.f).map_err(e("upload f"))?;
        let mut stat = s
            .alloc_zeros::<f64>(2 * batch.len().max(1))
            .map_err(e("alloc stat"))?;
        let mut scratch = s
            .alloc_zeros::<f64>(batch.scratch_len())
            .map_err(e("alloc scratch"))?;
        let (nmax, newton, mj, coupling, nse) = (
            batch.nmax as u32,
            cfg.newton as u32,
            cfg.mujoco_compat as u32,
            if cfg.block_diagonal { 2u32 } else { 0 },
            cfg.no_stall_exit as u32,
        );
        let lc = LaunchConfig {
            grid_dim: (nprob.div_ceil(block).max(1), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        s.synchronize().map_err(e("sync"))?;
        let t = std::time::Instant::now();
        // SAFETY: buffers sized by `ConvexBatch` for `nprob` problems at `nmax`.
        unsafe {
            s.launch_builder(&self.func)
                .arg(&nprob)
                .arg(&nmax)
                .arg(&cfg.regularization)
                .arg(&cfg.tolerance)
                .arg(&cfg.relaxation)
                .arg(&cfg.max_iterations)
                .arg(&newton)
                .arg(&mj)
                .arg(&coupling)
                .arg(&nse)
                .arg(&ncont)
                .arg(&a)
                .arg(&b)
                .arg(&rows)
                .arg(&mut f)
                .arg(&mut stat)
                .arg(&mut scratch)
                .launch(lc)
        }
        .map_err(e("launch phyz_convex_solve"))?;
        s.synchronize().map_err(e("sync"))?;
        let secs = t.elapsed().as_secs_f64();
        batch.f = s.clone_dtoh(&f).map_err(e("download f"))?;
        let st = s.clone_dtoh(&stat).map_err(e("download stat"))?;
        Ok((stats(&st[..2 * batch.len()]), secs))
    }
}
