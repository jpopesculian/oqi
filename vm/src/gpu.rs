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
//! caller drives it with a runtime's `block_on`, e.g. tokio). The only
//! target-specific bit is the readback: natively we drive the GPU with
//! `device.poll(Wait)`, while on wasm the browser event loop fulfils the
//! buffer map.

use std::borrow::Cow;

use async_trait::async_trait;
use num_complex::Complex;
use oqi_quantum::{Gate, Unitary};
use wgpu::util::DeviceExt;

use crate::backend::{GateModifiers, QuantumBackend};
use crate::sim::Rng;

const WORKGROUP_SIZE: u32 = 64;
/// `dispatch_workgroups` is capped at 65535 per dimension; tile across the
/// y dimension when the pair count needs more groups than that in x.
const MAX_DIM: u32 = 65535;
const DEFAULT_SEED: u64 = 0x2545F4914F6CDD1D;

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
    /// Create a GPU simulator with `num_qubits` qubits in |0…0⟩, or an error
    /// string if no adapter/device is available or the state can't be
    /// allocated. Async: `.await` it (native callers can block on it with a
    /// runtime, e.g. `tokio`; the browser awaits it directly).
    pub async fn new(num_qubits: u32) -> Result<Self, String> {
        let len = 1usize
            .checked_shl(num_qubits)
            .ok_or_else(|| format!("{num_qubits} qubits: state vector length overflows usize"))?;

        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .map_err(|e| format!("no suitable GPU adapter found: {e}"))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("oqi-gpu"),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("failed to create GPU device: {e}"))?;

        // Initialise |0…0⟩ on the host and upload.
        let mut init = vec![0f32; len * 2];
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
            rng: Rng::new(DEFAULT_SEED),
        })
    }

    /// Reseed the measurement RNG so runs are reproducible (defaults to
    /// [`DEFAULT_SEED`] otherwise).
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

    fn apply_x(&mut self, target: u32) {
        // U(π, 0, π) is exactly Pauli-X (no spurious phase).
        let x = Unitary::<f32>::new(std::f32::consts::PI, 0.0, std::f32::consts::PI).matrix();
        self.run_gate(&x, target, &[], &[]);
    }

    /// Read the full state back to the host as raw amplitudes (the tracked
    /// global phase is not applied).
    ///
    /// Async: the buffer map is awaited rather than blocked on. Natively we
    /// drive the GPU with `device.poll(Wait)` so the callback fires before we
    /// await; in the browser `poll` is a no-op and the event loop fulfils the
    /// map, so the `.await` suspends until then.
    async fn read_state(&self) -> Vec<Complex<f64>> {
        let bytes = (self.len * 2 * std::mem::size_of::<f32>()) as u64;
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

        let data = slice.get_mapped_range();
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

#[async_trait(?Send)]
impl QuantumBackend for GpuSim {
    async fn u(&mut self, target: u32, theta: f64, phi: f64, lambda: f64, m: &GateModifiers) {
        let unitary = Unitary::<f32>::new(theta as f32, phi as f32, lambda as f32);
        let mut g = Gate::new(unitary);
        if m.power != 1.0 {
            g = g.pow(m.power as f32);
        }
        let mat = g.matrix();
        self.run_gate(&mat, target, &m.controls, &m.neg_controls);
    }

    async fn gphase(&mut self, gamma: f64, m: &GateModifiers) {
        let g = gamma * m.power;

        if m.controls.is_empty() && m.neg_controls.is_empty() {
            self.global_phase += g;
            return;
        }

        // Controlled global phase is a relative phase on the innermost
        // control: diag(1, e^{ig}); a negctrl target is wrapped in X.
        let mut controls = m.controls.clone();
        let mut neg_controls = m.neg_controls.clone();
        let (target, neg_target) = match controls.pop() {
            Some(c) => (c, false),
            None => (neg_controls.pop().expect("at least one control"), true),
        };

        let mat = Unitary::<f32>::new(0.0, 0.0, g as f32).matrix();
        if neg_target {
            self.apply_x(target);
            self.run_gate(&mat, target, &controls, &neg_controls);
            self.apply_x(target);
        } else {
            self.run_gate(&mat, target, &controls, &neg_controls);
        }
    }

    async fn measure(&mut self, qubit: u32) -> bool {
        let bit = 1usize << qubit;
        let mut amps = self.read_state().await;

        let p_one: f64 = amps
            .iter()
            .enumerate()
            .filter(|(i, _)| i & bit != 0)
            .map(|(_, a)| a.norm_sqr())
            .sum();

        let outcome = self.rng.next_f64() < p_one;

        let norm = if outcome { p_one } else { 1.0 - p_one };
        let scale = if norm > 0.0 { 1.0 / norm.sqrt() } else { 0.0 };
        for (i, a) in amps.iter_mut().enumerate() {
            if (i & bit != 0) == outcome {
                *a *= scale;
            } else {
                *a = Complex::new(0.0, 0.0);
            }
        }
        self.write_state(&amps);
        outcome
    }

    async fn reset(&mut self, qubit: u32) {
        if self.measure(qubit).await {
            self.apply_x(qubit);
        }
    }

    async fn reset_state(&mut self, _num_qubits: u32) {
        // Re-upload |0…0⟩ to the resident buffer, reusing the device/pipeline
        // (no new adapter). Leaves the RNG stream intact so shots stay
        // independent and reproducible.
        let mut init = vec![0f32; self.len * 2];
        init[0] = 1.0;
        self.queue
            .write_buffer(&self.state, 0, bytemuck::cast_slice(&init));
        self.queue.submit(std::iter::empty());
        self.global_phase = 0.0;
    }

    async fn amplitudes(&self) -> Option<Vec<Complex<f64>>> {
        let phase = Complex::from_polar(1.0, self.global_phase);
        Some(self.read_state().await.iter().map(|a| a * phase).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::StateVectorSim;

    async fn gates(b: &mut dyn QuantumBackend) {
        let plain = GateModifiers::none();
        for q in 0..5 {
            b.u(
                q,
                std::f64::consts::FRAC_PI_2,
                0.0,
                std::f64::consts::PI,
                &plain,
            )
            .await; // H
        }
        for q in 0..5 {
            b.u(q, 0.37, 0.11, 0.59, &plain).await; // arbitrary rotation
        }
        // CX(0->4) and a negctrl gate to exercise both masks.
        let c0 = GateModifiers {
            controls: vec![0],
            neg_controls: vec![],
            power: 1.0,
        };
        b.u(4, std::f64::consts::PI, 0.0, std::f64::consts::PI, &c0)
            .await;
        let nc = GateModifiers {
            controls: vec![],
            neg_controls: vec![2],
            power: 1.0,
        };
        b.u(3, std::f64::consts::PI, 0.0, std::f64::consts::PI, &nc)
            .await;
        // controlled + global phase
        b.gphase(0.7, &c0).await;
        b.gphase(0.3, &plain).await;
    }

    /// The GPU backend must agree with the reference scalar simulator to
    /// within single-precision tolerance. Skips gracefully when no GPU
    /// adapter is available (e.g. headless CI) rather than failing.
    #[tokio::test(flavor = "multi_thread")]
    async fn matches_scalar_within_f32_tolerance() {
        let mut gpu = match GpuSim::new(5).await {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping gpu test: {e}");
                return;
            }
        };
        let mut cpu = StateVectorSim::<f64>::try_zeroed(5, 0).unwrap();

        gates(&mut gpu).await;
        gates(&mut cpu).await;

        let a = gpu.amplitudes().await.unwrap();
        let b = cpu.amplitudes().await.unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).norm() < 1e-4, "gpu {x} vs scalar {y}");
        }
    }
}
