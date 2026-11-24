//! A WebGPU (`wgpu`) state-vector simulator backend.
//!
//! The state vector lives in a GPU storage buffer of `vec2<f32>` (real,
//! imaginary) and stays resident there across gates — each gate is one
//! compute dispatch with no host transfer. A single kernel handles both
//! single-qubit and controlled gates: every thread owns one amplitude pair
//! and tests the control mask itself, so no scalar fallback is needed.
//!
//! Single precision only: the WebGPU/WGSL spec has no `f64`, so this
//! backend is inherently `f32`. The global phase is tracked on the host and
//! resolved when amplitudes are read out.
//!
//! Measurement reads the state back, samples and collapses on the host, and
//! re-uploads — gates (the hot path) stay on the GPU, while a measurement
//! costs one readback plus one upload.
//!
//! All device operations are `async`: [`GpuSim::new`] and the buffer
//! readback `.await` rather than block, so the backend works on the browser
//! main thread (where blocking is illegal) as well as natively (where a
//! caller drives it with any `block_on`, e.g. `pollster`). The only
//! target-specific bit is the readback: natively we drive the GPU with
//! `device.poll(Wait)`, while on wasm the browser event loop fulfils the
//! buffer map.

use std::borrow::Cow;

use crate::state::{Gate, Unitary};
use async_trait::async_trait;
use num_complex::Complex;
use wgpu::util::DeviceExt;

use crate::backend::{GateModifiers, QuantumBackend};
use crate::error::{Error, ErrorKind, Result};
use crate::measure;
use crate::rng::Rng;

const WORKGROUP_SIZE: u32 = 64;
/// Bytes per amplitude in the state buffer: a `vec2<f32>`.
const STATE_ELEMENT_BYTES: u64 = 8;
/// `dispatch_workgroups` is capped at 65535 per dimension; tile across the
/// y dimension when the pair count needs more groups than that in x.
const MAX_DIM: u32 = 65535;

/// Per-gate uniform: the 2×2 matrix packed as two `vec4` rows, plus the
/// target bit, control masks, pair count, and dispatch width. Padded to a
/// 16-byte multiple for uniform-buffer layout.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    row0: [f32; 4], // (m00.re, m00.im, m01.re, m01.im)
    row1: [f32; 4], // (m10.re, m10.im, m11.re, m11.im)
    target_bit: u32,
    control_mask: u32,
    neg_mask: u32,
    num_pairs: u32,
    dispatch_width: u32,
    _pad: [u32; 3],
}

const SHADER: &str = r#"
struct Params {
    row0: vec4<f32>,
    row1: vec4<f32>,
    target_bit: u32,
    control_mask: u32,
    neg_mask: u32,
    num_pairs: u32,
    dispatch_width: u32,
};

@group(0) @binding(0) var<storage, read_write> amps: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> p: Params;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(64)
fn apply_gate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair = gid.x + gid.y * p.dispatch_width;
    if (pair >= p.num_pairs) {
        return;
    }
    let t = p.target_bit;
    // Insert a 0 bit at the target position to get the clear-target index.
    let lo = pair & (t - 1u);
    let hi = (pair & ~(t - 1u)) << 1u;
    let i = hi | lo;
    let j = i | t;
    if ((i & p.control_mask) != p.control_mask) {
        return;
    }
    if ((i & p.neg_mask) != 0u) {
        return;
    }
    let a = amps[i];
    let b = amps[j];
    amps[i] = cmul(p.row0.xy, a) + cmul(p.row0.zw, b);
    amps[j] = cmul(p.row1.xy, a) + cmul(p.row1.zw, b);
}
"#;

/// A `wgpu`-backed state-vector simulator (single precision).
///
/// The state stays in GPU memory across gates (one compute dispatch per
/// gate); a measurement reads it back, samples and collapses on the host,
/// and uploads it again. WGSL has no `f64`, so amplitudes are `f32`. The
/// constructor and readback are `async` so the backend works on the
/// browser's main thread; natively, block on them (e.g. with `pollster`).
/// At most 32 qubits, and only as many as one device buffer holds.
pub struct GpuSim {
    device: wgpu::Device,
    queue: wgpu::Queue,
    state: wgpu::Buffer,
    params: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    len: usize,
    global_phase: f64,
    rng: Rng,
}

impl GpuSim {
    /// A simulator of `num_qubits` qubits in |0…0⟩ on the default GPU
    /// adapter, measuring with the [`DEFAULT_SEED`](crate::DEFAULT_SEED).
    /// Async, because acquiring a device is: native callers can block on it
    /// (e.g. with `pollster`), the browser awaits it.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::BackendUnavailable`] if there is no usable adapter or
    /// device, and [`ErrorKind::TooManyQubits`] if the state vector doesn't
    /// fit in one of the device's buffers (or needs more than 32 qubits).
    pub async fn new(num_qubits: u32) -> Result<Self> {
        let unavailable = |why: String| Error::new(ErrorKind::BackendUnavailable(why));
        let too_many = || {
            Error::new(ErrorKind::TooManyQubits {
                requested: num_qubits,
            })
        };

        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .map_err(|e| unavailable(format!("no suitable GPU adapter found: {e}")))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("oqi-gpu"),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .map_err(|e| unavailable(format!("failed to create GPU device: {e}")))?;

        // The shader indexes amplitudes with `u32`, the whole state lives in
        // one storage buffer (`vec2<f32>` per amplitude), and the host
        // mirrors that buffer for uploads and readback.
        let limits = device.limits();
        let len = (num_qubits <= u32::BITS)
            .then(|| 1u64 << num_qubits)
            .filter(|len| {
                let bytes = len * STATE_ELEMENT_BYTES;
                bytes <= limits.max_buffer_size
                    && bytes <= limits.max_storage_buffer_binding_size
                    && usize::try_from(bytes).is_ok()
            })
            .ok_or_else(too_many)? as usize;

        // Initialise |0…0⟩ on the host and upload.
        let mut init: Vec<f32> = Vec::new();
        init.try_reserve_exact(len * 2).map_err(|_| too_many())?;
        init.resize(len * 2, 0.0);
        init[0] = 1.0;
        let state = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("state"),
            contents: bytemuck::cast_slice(&init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<GpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gate"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gate-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gate-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gate-pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("apply_gate"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gate-bind-group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
                },
            ],
        });

        Ok(GpuSim {
            device,
            queue,
            state,
            params,
            pipeline,
            bind_group,
            len,
            global_phase: 0.0,
            rng: Rng::new(crate::DEFAULT_SEED),
        })
    }

    /// Seed the measurement RNG: a given seed reproduces a run.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Rng::new(seed);
        self
    }

    /// Dispatch one gate: write the uniform, encode a compute pass, submit.
    fn run_gate(&mut self, m: &[[Complex<f32>; 2]; 2], target: u32, controls: &[u32], neg: &[u32]) {
        let num_pairs = (self.len / 2) as u32;
        let total_groups = num_pairs.div_ceil(WORKGROUP_SIZE);
        let groups_x = total_groups.clamp(1, MAX_DIM);
        let groups_y = total_groups.div_ceil(MAX_DIM).max(1);

        let mut control_mask = 0u32;
        for &c in controls {
            control_mask |= 1u32 << c;
        }
        let mut neg_mask = 0u32;
        for &c in neg {
            neg_mask |= 1u32 << c;
        }

        let params = GpuParams {
            row0: [m[0][0].re, m[0][0].im, m[0][1].re, m[0][1].im],
            row1: [m[1][0].re, m[1][0].im, m[1][1].re, m[1][1].im],
            target_bit: 1u32 << target,
            control_mask,
            neg_mask,
            num_pairs,
            dispatch_width: groups_x * WORKGROUP_SIZE,
            _pad: [0; 3],
        };
        self.queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Read the full state back to the host as raw amplitudes (the tracked
    /// global phase is not applied).
    ///
    /// Async: the buffer map is awaited rather than blocked on. Natively we
    /// drive the GPU with `device.poll(Wait)` so the callback fires before we
    /// await; in the browser `poll` is a no-op and the event loop fulfils the
    /// map, so the `.await` suspends until then.
    async fn read_state(&self) -> Vec<Complex<f64>> {
        let bytes = self.len as u64 * STATE_ELEMENT_BYTES;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&self.state, 0, &staging, 0, bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        }
        rx.await
            .expect("map_async callback dropped")
            .expect("buffer map failed");

        let data = slice.get_mapped_range().expect("buffer mapped just above");
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let out = (0..self.len)
            .map(|k| Complex::new(floats[2 * k] as f64, floats[2 * k + 1] as f64))
            .collect();
        drop(data);
        staging.unmap();
        out
    }

    /// Overwrite the GPU state with host amplitudes (used by measurement).
    fn write_state(&self, amps: &[Complex<f64>]) {
        let mut buf = vec![0f32; self.len * 2];
        for (k, a) in amps.iter().enumerate() {
            buf[2 * k] = a.re as f32;
            buf[2 * k + 1] = a.im as f32;
        }
        self.queue
            .write_buffer(&self.state, 0, bytemuck::cast_slice(&buf));
        self.queue.submit(std::iter::empty());
    }
}

impl std::fmt::Debug for GpuSim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuSim")
            .field("num_qubits", &self.len.trailing_zeros())
            .finish_non_exhaustive()
    }
}

#[async_trait(?Send)]
impl QuantumBackend for GpuSim {
    fn num_qubits(&self) -> u32 {
        self.len.trailing_zeros()
    }

    fn u(
        &mut self,
        target: u32,
        theta: f64,
        phi: f64,
        lambda: f64,
        m: &GateModifiers,
    ) -> Result<(), ErrorKind> {
        let unitary = Unitary::<f32>::new(theta as f32, phi as f32, lambda as f32);
        let mut g = Gate::new(unitary);
        if m.power != 1.0 {
            g = g.pow(m.power as f32);
        }
        let mat = g.matrix();
        self.run_gate(&mat, target, &m.controls, &m.neg_controls);
        Ok(())
    }

    fn gphase(&mut self, gamma: f64) -> Result<(), ErrorKind> {
        self.global_phase += gamma;
        Ok(())
    }

    async fn measure(&mut self, qubit: u32) -> Result<bool, ErrorKind> {
        Ok(self.measure_all(&[qubit]).await?[0])
    }

    async fn measure_all(&mut self, qubits: &[u32]) -> Result<Vec<bool>, ErrorKind> {
        // One readback for the whole register: sample and collapse each qubit
        // on the host copy in turn (equivalent to sequential `measure`), then
        // one upload. Turns a per-qubit GPU round-trip into a single pair.
        let mut amps = self.read_state().await;
        let outcomes = qubits
            .iter()
            .map(|&q| measure::measure(&mut amps, q, &mut self.rng))
            .collect();
        self.write_state(&amps);
        Ok(outcomes)
    }

    async fn reset(&mut self, qubit: u32) -> Result<(), ErrorKind> {
        if self.measure(qubit).await? {
            // U(π, 0, π) is i·X: Pauli-X up to a global phase of i.
            let x = Unitary::<f32>::new(std::f32::consts::PI, 0.0, std::f32::consts::PI).matrix();
            self.run_gate(&x, qubit, &[], &[]);
        }
        Ok(())
    }

    fn reset_state(&mut self) -> Result<(), ErrorKind> {
        // Re-upload |0…0⟩ to the resident buffer, reusing the device/pipeline
        // (no new adapter).
        let mut init = vec![0f32; self.len * 2];
        init[0] = 1.0;
        self.queue
            .write_buffer(&self.state, 0, bytemuck::cast_slice(&init));
        self.queue.submit(std::iter::empty());
        self.global_phase = 0.0;
        Ok(())
    }

    async fn amplitudes(&self) -> Option<Vec<Complex<f64>>> {
        let phase = Complex::from_polar(1.0, self.global_phase);
        Some(self.read_state().await.iter().map(|a| a * phase).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::apply_gphase;
    use crate::sim::StateVectorSim;
    use std::f64::consts::{FRAC_PI_2, PI};

    fn gates(b: &mut dyn QuantumBackend) {
        let plain = GateModifiers::none();
        for q in 0..5 {
            b.u(q, FRAC_PI_2, 0.0, PI, &plain).unwrap(); // H
        }
        for q in 0..5 {
            b.u(q, 0.37, 0.11, 0.59, &plain).unwrap(); // arbitrary rotation
        }
        // CX(0->4) and a negctrl gate to exercise both masks.
        let c0 = GateModifiers::ctrl(&[0]);
        b.u(4, PI, 0.0, PI, &c0).unwrap();
        let nc = GateModifiers {
            neg_controls: vec![2],
            ..GateModifiers::none()
        };
        b.u(3, PI, 0.0, PI, &nc).unwrap();
        // controlled + global phase
        apply_gphase(b, 0.7, &c0).unwrap();
        b.gphase(0.3).unwrap();
    }

    /// A simulator, or `None` (with a note) when no GPU adapter is available
    /// (e.g. headless CI), so tests skip gracefully instead of failing.
    fn gpu(num_qubits: u32) -> Option<Result<GpuSim>> {
        let _turn = crate::spec::gpu_turn();
        match pollster::block_on(GpuSim::new(num_qubits)) {
            Err(Error {
                kind: ErrorKind::BackendUnavailable(why),
                ..
            }) => {
                eprintln!("skipping gpu test: {why}");
                None
            }
            sim => Some(sim),
        }
    }

    /// The GPU backend must agree with the reference scalar simulator to
    /// within single-precision tolerance.
    #[test]
    fn matches_scalar_within_f32_tolerance() {
        pollster::block_on(async {
            let Some(gpu) = gpu(5) else { return };
            let mut gpu = gpu.unwrap();
            let mut cpu = StateVectorSim::<f64>::new(5)
                .map(|s| s.with_seed(0))
                .unwrap();

            gates(&mut gpu);
            gates(&mut cpu);

            let a = gpu.amplitudes().await.unwrap();
            let b = cpu.amplitudes().await.unwrap();
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(&b) {
                assert!((x - y).norm() < 1e-4, "gpu {x} vs scalar {y}");
            }
        })
    }

    /// A state vector that can't fit in a device buffer is a clean error,
    /// not a host allocation failure or a device validation panic.
    #[test]
    fn oversized_state_is_too_many_qubits() {
        pollster::block_on(async {
            for n in [40, 64, 200] {
                let Some(gpu) = gpu(n) else { return };
                assert!(matches!(
                    gpu.err().map(|e| e.kind),
                    Some(ErrorKind::TooManyQubits { requested }) if requested == n
                ));
            }
        })
    }
}
