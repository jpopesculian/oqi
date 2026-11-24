//! A WebGPU (`wgpu`) state-vector simulator backend.
//!
//! The state vector lives in GPU storage buffers of `vec2<f32>` (real,
//! imaginary) and stays resident there: gates, measurement, reset and shot
//! sampling all run as compute dispatches, and only small results (partial
//! sums, the blocks a shot lands in) come back to the host.
//!
//! # Chunks
//!
//! A device binds at most `max_storage_buffer_binding_size` bytes of one
//! buffer (2 GiB on many GPUs, often less in browsers), so the 2ⁿ amplitudes
//! are split into 2ᵏ equal *chunks*, chunk `c` holding the amplitudes whose
//! top `k` index bits are `c`. Each chunk is its own buffer; the state is
//! one chunk whenever it fits in one binding. A gate whose target qubit is
//! inside a chunk runs once per chunk; one whose target is a chunk bit pairs
//! chunk `a` with chunk `a | bit` and runs once per pair. Every dispatch
//! binds at most two chunks, well inside WebGPU's default of eight storage
//! buffers per shader stage.
//!
//! # Measurement
//!
//! Measuring qubit `q` is the same computation as on the CPU simulators: a
//! reduction kernel sums |a|² over the amplitudes with bit `q` set into one
//! partial per workgroup, the host adds the partials in `f64` to get
//! P(q = 1), draws `rng < P`, and a collapse kernel zeroes the other half and
//! renormalizes. A given seed therefore draws the same outcomes as
//! [`StateVectorSim`](crate::StateVectorSim) (up to single-precision
//! rounding of P).
//!
//! Single precision only: the WebGPU/WGSL spec has no `f64`. The global
//! phase is tracked on the host and applied when amplitudes are read out.
//!
//! All device operations are `async`: [`GpuSim::new`] and every readback
//! `.await` rather than block, so the backend works on the browser main
//! thread (where blocking is illegal) as well as natively (where a caller
//! drives it with any `block_on`, e.g. `pollster`). The only target-specific
//! bit is the readback: natively we drive the GPU with `device.poll(Wait)`,
//! while on wasm the browser event loop fulfils the buffer map.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use num_complex::Complex;

use crate::backend::{GateModifiers, MAX_STATE_BYTES, QuantumBackend};
use crate::error::{Error, ErrorKind, Result};
use crate::measure;
use crate::rng::Rng;
use crate::state::{Gate, Unitary};

/// Threads per workgroup, in every kernel.
const WORKGROUP_SIZE: u32 = 64;
/// Amplitudes each thread of the reduction kernel sums.
const PER_THREAD: u32 = 256;
/// Amplitudes one reduction workgroup covers: one partial sum each.
const BLOCK_AMPS: u64 = (WORKGROUP_SIZE * PER_THREAD) as u64;
/// Bytes per amplitude in a state buffer: a `vec2<f32>`.
const AMP_BYTES: u64 = 8;
/// `dispatch_workgroups` is capped at 65535 per dimension; tile across the
/// y dimension when a dispatch needs more groups than that in x.
const MAX_DIM: u32 = 65535;
/// Amplitude indices are `u32` in the kernels.
const MAX_QUBITS: u32 = u32::BITS;
/// The most a shot-sampling readback copies in one go.
const READBACK_BATCH_BYTES: u64 = 64 << 20;

/// What the kernels do with a dispatch's chunk(s).
const MODE_WITHIN_CHUNK: u32 = 0;
const MODE_ACROSS_CHUNKS: u32 = 1;

/// One dispatch's uniform: the 2×2 matrix packed as two `vec4` rows, the
/// global index of the first amplitude of each bound chunk, the gate's
/// target bit and control masks, the measurement filter and scale, and the
/// sizes. Mirrors `Params` in [`SHADER`].
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    row0: [f32; 4], // (m00.re, m00.im, m01.re, m01.im)
    row1: [f32; 4], // (m10.re, m10.im, m11.re, m11.im)
    base_lo: u32,
    base_hi: u32,
    target_bit: u32,
    control_mask: u32,
    neg_mask: u32,
    count: u32,
    mode: u32,
    filter_mask: u32,
    filter_value: u32,
    partial_offset: u32,
    groups: u32,
    scale: f32,
}

const SHADER: &str = r#"
struct Params {
    row0: vec4<f32>,
    row1: vec4<f32>,
    base_lo: u32,
    base_hi: u32,
    target_bit: u32,
    control_mask: u32,
    neg_mask: u32,
    count: u32,
    mode: u32,
    filter_mask: u32,
    filter_value: u32,
    partial_offset: u32,
    groups: u32,
    scale: f32,
};

// `lo` is the chunk being worked on; `hi` the chunk paired with it when a
// gate's target bit is a chunk bit (otherwise a one-element placeholder:
// two writable bindings may not alias one buffer).
@group(0) @binding(0) var<storage, read_write> lo: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> hi: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var<storage, read_write> partials: array<f32>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn thread_index(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.x + gid.y * nwg.x * 64u;
}

fn controls_hold(g: u32) -> bool {
    return (g & p.control_mask) == p.control_mask && (g & p.neg_mask) == 0u;
}

// One thread per amplitude pair.
@compute @workgroup_size(64)
fn apply_gate(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let k = thread_index(gid, nwg);
    if (k >= p.count) {
        return;
    }
    if (p.mode == 0u) {
        // Both amplitudes in `lo`: insert a 0 at the target bit.
        let t = p.target_bit;
        let i = ((k & ~(t - 1u)) << 1u) | (k & (t - 1u));
        let j = i | t;
        if (!controls_hold(p.base_lo | i)) {
            return;
        }
        let a = lo[i];
        let b = lo[j];
        lo[i] = cmul(p.row0.xy, a) + cmul(p.row0.zw, b);
        lo[j] = cmul(p.row1.xy, a) + cmul(p.row1.zw, b);
    } else {
        // Amplitude `k` of `lo` (target bit 0) pairs with `k` of `hi`.
        if (!controls_hold(p.base_lo | k)) {
            return;
        }
        let a = lo[k];
        let b = hi[k];
        lo[k] = cmul(p.row0.xy, a) + cmul(p.row0.zw, b);
        hi[k] = cmul(p.row1.xy, a) + cmul(p.row1.zw, b);
    }
}

// Sum |a|² over the amplitudes of `lo` whose global index matches the
// filter: each thread sums 256, the workgroup adds its threads' sums, and
// one partial per workgroup goes to `partials`.
var<workgroup> scratch: array<f32, 64>;

@compute @workgroup_size(64)
fn reduce(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let group = wid.x + wid.y * nwg.x;
    let start = group * 64u * 256u + lid.x;
    var sum = 0.0;
    for (var m = 0u; m < 256u; m = m + 1u) {
        let k = start + m * 64u;
        if (group < p.groups && k < p.count
            && ((p.base_lo | k) & p.filter_mask) == p.filter_value) {
            let a = lo[k];
            sum = sum + dot(a, a);
        }
    }
    scratch[lid.x] = sum;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
        if (lid.x < s) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s];
        }
        workgroupBarrier();
    }
    if (lid.x == 0u && group < p.groups) {
        partials[p.partial_offset + group] = scratch[0];
    }
}

// Keep the amplitudes of `lo` whose global index matches the filter, scaled;
// zero the rest. One thread per amplitude.
@compute @workgroup_size(64)
fn collapse(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let k = thread_index(gid, nwg);
    if (k >= p.count) {
        return;
    }
    if (((p.base_lo | k) & p.filter_mask) == p.filter_value) {
        lo[k] = lo[k] * p.scale;
    } else {
        lo[k] = vec2<f32>(0.0, 0.0);
    }
}
"#;

/// A `wgpu`-backed state-vector simulator (single precision).
///
/// The state stays in GPU memory: gates, measurement, reset and shot
/// sampling are compute dispatches, and only small results come back to the
/// host. WGSL has no `f64`, so amplitudes are `f32`; measurement
/// probabilities are summed in `f64` on the host, so a given seed draws the
/// same outcomes as [`StateVectorSim`](crate::StateVectorSim).
///
/// The state is split across as many device buffers as the adapter's
/// binding limit requires, so the qubit count is bounded by device memory
/// (and by 32, since the kernels index amplitudes with `u32`): 31 qubits
/// take 16 GiB. [`amplitudes`](QuantumBackend::amplitudes)
/// copies the state to the host and so returns `None` past 16 GiB of `f64`
/// amplitudes (30 qubits).
///
/// The constructor and readbacks are `async` so the backend works on the
/// browser's main thread; natively, block on them (e.g. with `pollster`).
pub struct GpuSim {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// The state: chunk `c` holds the amplitudes whose top index bits are
    /// `c`.
    chunks: Vec<wgpu::Buffer>,
    /// Per-dispatch [`GpuParams`], one every `param_stride` bytes, bound
    /// with a dynamic offset.
    params: wgpu::Buffer,
    param_stride: u64,
    /// One reduction partial per workgroup per chunk, and its readback copy.
    partials: wgpu::Buffer,
    partials_readback: wgpu::Buffer,
    gate: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    collapse: wgpu::ComputePipeline,
    /// Chunk `c` alone (with a placeholder as `hi`).
    single: Vec<wgpu::BindGroup>,
    /// `pairs[h][i]`: the `i`-th chunk with chunk bit `h` clear, and its
    /// partner with the bit set.
    pairs: Vec<Vec<wgpu::BindGroup>>,
    num_qubits: u32,
    /// log₂ of the amplitudes per chunk.
    chunk_bits: u32,
    global_phase: f64,
    rng: Rng,
    /// The first device error reported outside a call (wgpu reports them
    /// asynchronously); the next operation returns it.
    device_error: Arc<Mutex<Option<String>>>,
}

/// One dispatch: its uniform, its chunk binding and its workgroup count.
struct Dispatch<'a> {
    params: GpuParams,
    bind_group: &'a wgpu::BindGroup,
    groups: u32,
}

fn backend_error(why: impl Into<String>) -> ErrorKind {
    ErrorKind::Backend(why.into())
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
    /// device, and [`ErrorKind::TooManyQubits`] for more than 32 qubits or
    /// a state the device can't allocate.
    pub async fn new(num_qubits: u32) -> Result<Self> {
        Self::with_chunk_limit(num_qubits, None).await
    }

    /// [`new`](Self::new), with chunks of at most `max_chunk_amps`
    /// amplitudes (when smaller than the device allows): tests use it to
    /// exercise many chunks on small states.
    async fn with_chunk_limit(num_qubits: u32, max_chunk_amps: Option<u64>) -> Result<Self> {
        let unavailable = |why: String| Error::new(ErrorKind::BackendUnavailable(why));
        let too_many = || {
            Error::new(ErrorKind::TooManyQubits {
                requested: num_qubits,
            })
        };
        if num_qubits > MAX_QUBITS {
            return Err(too_many());
        }

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

        // wgpu reports most device errors asynchronously; record the first
        // instead of panicking, and return it from the next operation.
        let device_error = Arc::new(Mutex::new(None::<String>));
        let slot = device_error.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            if let Ok(mut slot) = slot.lock() {
                slot.get_or_insert_with(|| e.to_string());
            }
        }));

        // The largest power-of-two chunk one binding (and one buffer) holds.
        let limits = device.limits();
        let device_chunk = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size)
            / AMP_BYTES;
        let max_chunk = max_chunk_amps.map_or(device_chunk, |m| m.min(device_chunk));
        if max_chunk == 0 {
            return Err(too_many());
        }
        let chunk_bits = num_qubits.min(max_chunk.ilog2());
        let num_chunks = 1usize << (num_qubits - chunk_bits);
        let chunk_amps = 1u64 << chunk_bits;
        let groups_per_chunk = chunk_amps.div_ceil(BLOCK_AMPS);
        let total_partials = num_chunks as u64 * groups_per_chunk;

        let param_stride = (std::mem::size_of::<GpuParams>() as u64)
            .next_multiple_of(u64::from(limits.min_uniform_buffer_offset_alignment));

        // The state is the bulk of the allocation; device memory isn't
        // queryable, so an allocation the device refuses means too many
        // qubits. (New buffers are zeroed.)
        let scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let storage = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let state_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let chunks: Vec<wgpu::Buffer> = (0..num_chunks)
            .map(|_| storage("state chunk", chunk_amps * AMP_BYTES, state_usage))
            .collect();
        let placeholder = storage("placeholder", AMP_BYTES, wgpu::BufferUsages::STORAGE);
        let params = storage(
            "params",
            num_chunks as u64 * param_stride,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let partials = storage(
            "partials",
            total_partials * 4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let partials_readback = storage(
            "partials readback",
            total_partials * 4,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        if scope.pop().await.is_some() {
            return Err(too_many());
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("state kernels"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
        });
        let storage_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let params_size = NonZeroU64::new(std::mem::size_of::<GpuParams>() as u64);
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("state-bgl"),
            entries: &[
                storage_entry(0),
                storage_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: params_size,
                    },
                    count: None,
                },
                storage_entry(3),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("state-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (gate, reduce, collapse) = (
            pipeline("apply_gate"),
            pipeline("reduce"),
            pipeline("collapse"),
        );

        fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        let bind_group = |lo: &wgpu::Buffer, hi: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("state-bind-group"),
                layout: &bgl,
                entries: &[
                    entry(0, lo),
                    entry(1, hi),
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &params,
                            offset: 0,
                            size: params_size,
                        }),
                    },
                    entry(3, &partials),
                ],
            })
        };
        let single = chunks.iter().map(|c| bind_group(c, &placeholder)).collect();
        let pairs = (0..num_qubits - chunk_bits)
            .map(|h| {
                (0..num_chunks)
                    .filter(|a| a >> h & 1 == 0)
                    .map(|a| bind_group(&chunks[a], &chunks[a | 1 << h]))
                    .collect()
            })
            .collect();

        let sim = GpuSim {
            device,
            queue,
            chunks,
            params,
            param_stride,
            partials,
            partials_readback,
            gate,
            reduce,
            collapse,
            single,
            pairs,
            num_qubits,
            chunk_bits,
            global_phase: 0.0,
            rng: Rng::new(crate::DEFAULT_SEED),
            device_error,
        };
        sim.set_ground_amplitude();
        sim.check().map_err(Error::new)?;
        Ok(sim)
    }

    /// Seed the measurement RNG: a given seed reproduces a run.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Rng::new(seed);
        self
    }

    fn chunk_amps(&self) -> u64 {
        1 << self.chunk_bits
    }

    fn num_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// The global index of chunk `c`'s first amplitude.
    fn chunk_base(&self, c: usize) -> u32 {
        // At most 2³² amplitudes, so the base fits.
        ((c as u64) << self.chunk_bits) as u32
    }

    /// Workgroups of the reduction kernel per chunk (= partials per chunk).
    fn groups_per_chunk(&self) -> u32 {
        self.chunk_amps().div_ceil(BLOCK_AMPS) as u32
    }

    /// Return the first device error reported since the last check.
    fn check(&self) -> std::result::Result<(), ErrorKind> {
        match self.device_error.lock().ok().and_then(|mut e| e.take()) {
            Some(why) => Err(backend_error(format!("GPU device error: {why}"))),
            None => Ok(()),
        }
    }

    /// Write amplitude 0 = 1 (the rest of a fresh or cleared state is 0).
    fn set_ground_amplitude(&self) {
        self.queue
            .write_buffer(&self.chunks[0], 0, bytemuck::cast_slice(&[1.0f32, 0.0]));
        self.queue.submit(std::iter::empty());
    }

    /// Record `dispatches` of `pipeline` into `encoder`: their uniforms go
    /// into the params buffer in one write (so at most one `encode` per
    /// submit), each dispatch binding its slice with a dynamic offset.
    fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        dispatches: &[Dispatch<'_>],
    ) {
        if dispatches.is_empty() {
            return;
        }
        let stride = self.param_stride as usize;
        let mut bytes = vec![0u8; dispatches.len() * stride];
        for (slot, d) in bytes.chunks_mut(stride).zip(dispatches) {
            let params = bytemuck::bytes_of(&d.params);
            slot[..params.len()].copy_from_slice(params);
        }
        self.queue.write_buffer(&self.params, 0, &bytes);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        for (i, d) in dispatches.iter().enumerate() {
            let offset = (i as u64 * self.param_stride) as u32;
            pass.set_bind_group(0, d.bind_group, &[offset]);
            let groups = d.groups.max(1);
            pass.dispatch_workgroups(groups.min(MAX_DIM), groups.div_ceil(MAX_DIM), 1);
        }
    }

    fn encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default())
    }

    /// Whether chunk `c` can hold any index with `bits & mask == value`:
    /// only the chunk bits of the mask can rule a whole chunk out.
    fn chunk_may_match(&self, c: usize, mask: u32, value: u32) -> bool {
        let chunk_mask = !((self.chunk_amps() - 1) as u32);
        (self.chunk_base(c) & mask & chunk_mask) == (value & chunk_mask)
    }

    /// Dispatch one gate: one dispatch per chunk when the target bit is
    /// inside a chunk, one per chunk pair when it is a chunk bit; chunks
    /// the control masks rule out are skipped.
    fn run_gate(&self, m: &[[Complex<f32>; 2]; 2], target: u32, controls: &[u32], neg: &[u32]) {
        let control_mask = controls.iter().fold(0u32, |mask, &c| mask | 1 << c);
        let neg_mask = neg.iter().fold(0u32, |mask, &c| mask | 1 << c);
        let base = GpuParams {
            row0: [m[0][0].re, m[0][0].im, m[0][1].re, m[0][1].im],
            row1: [m[1][0].re, m[1][0].im, m[1][1].re, m[1][1].im],
            control_mask,
            neg_mask,
            ..GpuParams::default()
        };
        // Both masks as one filter: controls must be 1, negative ones 0.
        let (mask, value) = (control_mask | neg_mask, control_mask);
        let chunk_amps = self.chunk_amps() as u32;
        let threads = |count: u32| count.div_ceil(WORKGROUP_SIZE);

        let dispatches: Vec<Dispatch<'_>> = if target < self.chunk_bits {
            let count = chunk_amps / 2;
            (0..self.num_chunks())
                .filter(|&c| self.chunk_may_match(c, mask, value))
                .map(|c| Dispatch {
                    params: GpuParams {
                        base_lo: self.chunk_base(c),
                        target_bit: 1 << target,
                        count,
                        mode: MODE_WITHIN_CHUNK,
                        ..base
                    },
                    bind_group: &self.single[c],
                    groups: threads(count),
                })
                .collect()
        } else {
            let h = target - self.chunk_bits;
            (0..self.num_chunks())
                .filter(|a| a >> h & 1 == 0)
                .enumerate()
                .filter(|&(_, a)| self.chunk_may_match(a, mask, value))
                .map(|(i, a)| Dispatch {
                    params: GpuParams {
                        base_lo: self.chunk_base(a),
                        base_hi: self.chunk_base(a | 1 << h),
                        count: chunk_amps,
                        mode: MODE_ACROSS_CHUNKS,
                        ..base
                    },
                    bind_group: &self.pairs[h as usize][i],
                    groups: threads(chunk_amps),
                })
                .collect()
        };
        if dispatches.is_empty() {
            return;
        }
        let mut encoder = self.encoder();
        self.encode(&mut encoder, &self.gate, &dispatches);
        self.queue.submit(Some(encoder.finish()));
    }

    /// Per workgroup of every chunk, the sum of |a|² over the amplitudes
    /// whose index has `bits & mask == value`: one partial per
    /// [`BLOCK_AMPS`] amplitudes (or per chunk, if chunks are smaller), in
    /// index order. Chunks the filter rules out are not read; their
    /// partials are 0.
    async fn partial_sums(
        &self,
        mask: u32,
        value: u32,
    ) -> std::result::Result<Vec<f32>, ErrorKind> {
        let gpc = self.groups_per_chunk();
        let count = self.chunk_amps() as u32;
        let read: Vec<usize> = (0..self.num_chunks())
            .filter(|&c| self.chunk_may_match(c, mask, value))
            .collect();
        let dispatches: Vec<Dispatch<'_>> = read
            .iter()
            .map(|&c| Dispatch {
                params: GpuParams {
                    base_lo: self.chunk_base(c),
                    count,
                    filter_mask: mask,
                    filter_value: value,
                    partial_offset: c as u32 * gpc,
                    groups: gpc,
                    ..GpuParams::default()
                },
                bind_group: &self.single[c],
                groups: gpc,
            })
            .collect();
        let mut encoder = self.encoder();
        self.encode(&mut encoder, &self.reduce, &dispatches);
        let bytes = self.partials.size();
        encoder.copy_buffer_to_buffer(&self.partials, 0, &self.partials_readback, 0, bytes);
        self.queue.submit(Some(encoder.finish()));

        let data = self.map_read(&self.partials_readback, bytes).await?;
        let all: &[f32] = bytemuck::cast_slice(&data);
        let mut sums = vec![0.0f32; all.len()];
        let gpc = gpc as usize;
        for c in read {
            sums[c * gpc..(c + 1) * gpc].copy_from_slice(&all[c * gpc..(c + 1) * gpc]);
        }
        Ok(sums)
    }

    /// Keep the amplitudes whose index has `bits & mask == value`, times
    /// `scale`, and zero the rest. A chunk the filter rules out is cleared
    /// without a dispatch.
    fn collapse_onto(&self, mask: u32, value: u32, scale: f32) {
        let count = self.chunk_amps() as u32;
        let mut encoder = self.encoder();
        let mut dispatches = Vec::new();
        for c in 0..self.num_chunks() {
            if !self.chunk_may_match(c, mask, value) {
                encoder.clear_buffer(&self.chunks[c], 0, None);
                continue;
            }
            dispatches.push(Dispatch {
                params: GpuParams {
                    base_lo: self.chunk_base(c),
                    count,
                    filter_mask: mask,
                    filter_value: value,
                    scale,
                    ..GpuParams::default()
                },
                bind_group: &self.single[c],
                groups: count.div_ceil(WORKGROUP_SIZE),
            });
        }
        self.encode(&mut encoder, &self.collapse, &dispatches);
        self.queue.submit(Some(encoder.finish()));
    }

    /// Map `buffer` (a `MAP_READ` copy target, already filled by submitted
    /// work) and copy out its first `size` bytes.
    ///
    /// Async: the buffer map is awaited rather than blocked on. Natively we
    /// drive the GPU with `device.poll(Wait)` so the callback fires before we
    /// await; in the browser `poll` is a no-op and the event loop fulfils the
    /// map, so the `.await` suspends until then.
    async fn map_read(
        &self,
        buffer: &wgpu::Buffer,
        size: u64,
    ) -> std::result::Result<Vec<u8>, ErrorKind> {
        let slice = buffer.slice(..size);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        }
        rx.await
            .map_err(|_| backend_error("the GPU buffer map was dropped"))?
            .map_err(|e| backend_error(format!("GPU buffer map failed: {e}")))?;
        let data = slice
            .get_mapped_range()
            .map_err(|e| backend_error(format!("GPU buffer map failed: {e}")))?
            .to_vec();
        buffer.unmap();
        self.check()?;
        Ok(data)
    }

    /// A `MAP_READ` buffer of `size` bytes to copy results into.
    fn readback_buffer(&self, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// The raw amplitudes (without the global phase), chunk by chunk.
    async fn read_state(&self) -> std::result::Result<Vec<Complex<f64>>, ErrorKind> {
        let len = 1usize << self.num_qubits;
        let mut out = Vec::new();
        out.try_reserve_exact(len)
            .map_err(|_| backend_error("not enough host memory for the amplitudes"))?;
        let bytes = self.chunk_amps() * AMP_BYTES;
        let readback = self.readback_buffer(bytes);
        for chunk in &self.chunks {
            let mut encoder = self.encoder();
            encoder.copy_buffer_to_buffer(chunk, 0, &readback, 0, bytes);
            self.queue.submit(Some(encoder.finish()));
            let data = self.map_read(&readback, bytes).await?;
            let amps: &[[f32; 2]] = bytemuck::cast_slice(&data);
            out.extend(
                amps.iter()
                    .map(|&[re, im]| Complex::new(f64::from(re), f64::from(im))),
            );
        }
        Ok(out)
    }
}

impl std::fmt::Debug for GpuSim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuSim")
            .field("num_qubits", &self.num_qubits)
            .field("chunks", &self.num_chunks())
            .finish_non_exhaustive()
    }
}

#[async_trait(?Send)]
impl QuantumBackend for GpuSim {
    fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    fn u(
        &mut self,
        target: u32,
        theta: f64,
        phi: f64,
        lambda: f64,
        m: &GateModifiers,
    ) -> std::result::Result<(), ErrorKind> {
        let unitary = Unitary::<f32>::new(theta as f32, phi as f32, lambda as f32);
        let mut g = Gate::new(unitary);
        if m.power != 1.0 {
            g = g.pow(m.power as f32);
        }
        let mat = g.matrix();
        self.run_gate(&mat, target, &m.controls, &m.neg_controls);
        self.check()
    }

    fn gphase(&mut self, gamma: f64) -> std::result::Result<(), ErrorKind> {
        self.global_phase += gamma;
        Ok(())
    }

    async fn measure(&mut self, qubit: u32) -> std::result::Result<bool, ErrorKind> {
        Ok(self.measure_all(&[qubit]).await?[0])
    }

    async fn measure_all(&mut self, qubits: &[u32]) -> std::result::Result<Vec<bool>, ErrorKind> {
        // As `measure::measure`, qubit by qubit: P(1) from the partial sums
        // (added in f64), one draw, then collapse on the device.
        let mut outcomes = Vec::with_capacity(qubits.len());
        for &q in qubits {
            let bit = 1u32 << q;
            let p_one: f64 = self
                .partial_sums(bit, bit)
                .await?
                .iter()
                .map(|&p| f64::from(p))
                .sum();
            let outcome = self.rng.next_f64() < p_one;
            let scale = measure::renorm(p_one, outcome) as f32;
            self.collapse_onto(bit, if outcome { bit } else { 0 }, scale);
            outcomes.push(outcome);
        }
        self.check()?;
        Ok(outcomes)
    }

    async fn reset(&mut self, qubit: u32) -> std::result::Result<(), ErrorKind> {
        if self.measure(qubit).await? {
            // U(π, 0, π) is i·X: Pauli-X up to a global phase of i.
            let x = Unitary::<f32>::new(std::f32::consts::PI, 0.0, std::f32::consts::PI).matrix();
            self.run_gate(&x, qubit, &[], &[]);
        }
        self.check()
    }

    fn reset_state(&mut self) -> std::result::Result<(), ErrorKind> {
        // Clear every chunk on the device and set amplitude 0: no host copy
        // of the state.
        let mut encoder = self.encoder();
        for chunk in &self.chunks {
            encoder.clear_buffer(chunk, 0, None);
        }
        self.queue.submit(Some(encoder.finish()));
        self.set_ground_amplitude();
        self.global_phase = 0.0;
        self.check()
    }

    async fn amplitudes(&self) -> Option<Vec<Complex<f64>>> {
        let bytes = (1u64 << self.num_qubits) * std::mem::size_of::<Complex<f64>>() as u64;
        if bytes > MAX_STATE_BYTES {
            return None;
        }
        let phase = Complex::from_polar(1.0, self.global_phase);
        let state = self.read_state().await.ok()?;
        Some(state.into_iter().map(|a| a * phase).collect())
    }

    async fn sample_shots(
        &mut self,
        shots: u32,
        seed: u64,
    ) -> std::result::Result<Option<Vec<Vec<bool>>>, ErrorKind> {
        // Inverse CDF in two levels, without copying the state: the block
        // sums (one per reduction workgroup) pick each shot's block, then
        // only the blocks drawn are read back to pick the index within.
        let blocks = self.partial_sums(0, 0).await?;
        let block_len = self.chunk_amps().min(BLOCK_AMPS);
        let mut cumulative = Vec::with_capacity(blocks.len());
        let mut total = 0.0f64;
        for &b in &blocks {
            total += f64::from(b);
            cumulative.push(total);
        }
        let total = total.max(f64::MIN_POSITIVE);

        // Each shot's block and how far into it its draw lands.
        let mut rng = Rng::new(seed);
        let mut by_block: BTreeMap<usize, Vec<(usize, f64)>> = BTreeMap::new();
        for shot in 0..shots as usize {
            let u = rng.next_f64() * total;
            let block = cumulative.partition_point(|&c| c < u).min(blocks.len() - 1);
            let before = if block == 0 {
                0.0
            } else {
                cumulative[block - 1]
            };
            by_block.entry(block).or_default().push((shot, u - before));
        }

        let mut out = vec![Vec::new(); shots as usize];
        let block_bytes = block_len * AMP_BYTES;
        let per_batch = (READBACK_BATCH_BYTES / block_bytes).max(1) as usize;
        let drawn: Vec<_> = by_block.into_iter().collect();
        let blocks_per_chunk = self.chunk_amps() / block_len;
        for batch in drawn.chunks(per_batch) {
            let readback = self.readback_buffer(batch.len() as u64 * block_bytes);
            let mut encoder = self.encoder();
            for (slot, (block, _)) in batch.iter().enumerate() {
                let chunk = (*block as u64 / blocks_per_chunk) as usize;
                let offset = (*block as u64 % blocks_per_chunk) * block_bytes;
                encoder.copy_buffer_to_buffer(
                    &self.chunks[chunk],
                    offset,
                    &readback,
                    slot as u64 * block_bytes,
                    block_bytes,
                );
            }
            self.queue.submit(Some(encoder.finish()));
            let data = self
                .map_read(&readback, batch.len() as u64 * block_bytes)
                .await?;
            let all: &[[f32; 2]] = bytemuck::cast_slice(&data);
            for (slot, (block, draws)) in batch.iter().enumerate() {
                let len = block_len as usize;
                let amps = &all[slot * len..][..len];
                let mut within = Vec::with_capacity(len);
                let mut sum = 0.0f64;
                for &[re, im] in amps {
                    sum += f64::from(re) * f64::from(re) + f64::from(im) * f64::from(im);
                    within.push(sum);
                }
                // The device's f32 block sum and this f64 one differ by
                // rounding: rescale the draw onto the block's own total.
                let rescale = match f64::from(blocks[*block]) {
                    partial if partial > 0.0 => sum / partial,
                    _ => 1.0,
                };
                for &(shot, residual) in draws {
                    let r = residual * rescale;
                    let idx = within.partition_point(|&c| c < r).min(within.len() - 1);
                    let index = *block as u64 * block_len + idx as u64;
                    out[shot] = (0..self.num_qubits)
                        .map(|q| (index >> q) & 1 == 1)
                        .collect();
                }
            }
        }
        Ok(Some(out))
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

    /// Gates over 10 qubits that, split into chunks of 16 to 1024
    /// amplitudes, target qubits inside a chunk and chunk bits, with
    /// controls and negative controls on both sides of the split.
    fn wide_gates(b: &mut dyn QuantumBackend) {
        let plain = GateModifiers::none();
        for q in 0..10 {
            b.u(q, FRAC_PI_2, 0.0, PI, &plain).unwrap(); // H
            b.u(q, 0.1 + 0.07 * f64::from(q), 0.3, -0.2, &plain)
                .unwrap();
        }
        let modified = |controls: &[u32], neg: &[u32], power: f64| GateModifiers {
            controls: controls.to_vec(),
            neg_controls: neg.to_vec(),
            power,
        };
        b.u(9, PI, 0.0, PI, &modified(&[0], &[], 1.0)).unwrap(); // low → high
        b.u(1, PI, 0.0, PI, &modified(&[8], &[], 1.0)).unwrap(); // high → low
        b.u(7, 0.4, 0.2, 0.9, &modified(&[2, 9], &[5], 1.0))
            .unwrap();
        b.u(3, 0.8, -0.1, 0.3, &modified(&[6], &[8], 1.0)).unwrap();
        b.u(8, 1.1, 0.5, 0.2, &modified(&[], &[4, 9], 1.0)).unwrap();
        b.u(6, 0.7, 0.1, 0.4, &modified(&[], &[], 0.5)).unwrap(); // pow
        b.u(9, 0.7, 0.1, 0.4, &modified(&[3], &[], -1.5)).unwrap();
        apply_gphase(b, 0.7, &modified(&[9], &[], 1.0)).unwrap();
        apply_gphase(b, -0.4, &modified(&[], &[2, 8], 1.0)).unwrap();
        b.gphase(0.3).unwrap();
    }

    /// Chunk sizes for 10 qubits: one chunk (the device's own limit), then
    /// 2, 4, 8, 16 and 64 chunks.
    const CHUNK_LIMITS: [Option<u64>; 6] =
        [None, Some(512), Some(256), Some(128), Some(64), Some(16)];

    /// A simulator, or `None` (with a note) when no GPU adapter is available
    /// (e.g. headless CI), so tests skip gracefully instead of failing.
    /// Callers hold [`crate::spec::gpu_turn`] for as long as they use the
    /// device.
    fn gpu_with(num_qubits: u32, chunk_limit: Option<u64>) -> Option<Result<GpuSim>> {
        match pollster::block_on(GpuSim::with_chunk_limit(num_qubits, chunk_limit)) {
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

    fn gpu(num_qubits: u32) -> Option<Result<GpuSim>> {
        gpu_with(num_qubits, None)
    }

    fn cpu(num_qubits: u32, seed: u64) -> StateVectorSim<f32> {
        StateVectorSim::<f32>::new(num_qubits)
            .map(|s| s.with_seed(seed))
            .unwrap()
    }

    async fn assert_same_state(gpu: &GpuSim, cpu: &dyn QuantumBackend, what: &str) {
        let a = gpu.amplitudes().await.unwrap();
        let b = cpu.amplitudes().await.unwrap();
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!((x - y).norm() < 1e-4, "{what}: |{i}⟩ gpu {x} vs cpu {y}");
        }
    }

    /// The GPU backend must agree with the reference scalar simulator to
    /// within single-precision tolerance.
    #[test]
    fn matches_scalar_within_f32_tolerance() {
        let _turn = crate::spec::gpu_turn();
        pollster::block_on(async {
            let Some(gpu) = gpu(5) else { return };
            let mut gpu = gpu.unwrap();
            let mut cpu = StateVectorSim::<f64>::new(5)
                .map(|s| s.with_seed(0))
                .unwrap();

            gates(&mut gpu);
            gates(&mut cpu);
            assert_same_state(&gpu, &cpu, "5 qubits").await;
        })
    }

    /// Splitting the state across chunk buffers changes nothing: every
    /// chunk count gives the scalar simulator's amplitudes.
    #[test]
    fn chunked_states_match_scalar() {
        let _turn = crate::spec::gpu_turn();
        pollster::block_on(async {
            for (limit, chunks) in CHUNK_LIMITS.into_iter().zip([1, 2, 4, 8, 16, 64]) {
                let Some(gpu) = gpu_with(10, limit) else {
                    return;
                };
                let mut gpu = gpu.unwrap();
                assert_eq!(gpu.num_chunks(), chunks);
                let mut cpu = cpu(10, 0);
                wide_gates(&mut gpu);
                wide_gates(&mut cpu);
                let what = format!("{} chunks", gpu.num_chunks());
                assert_same_state(&gpu, &cpu, &what).await;
            }
        })
    }

    /// Measured on the device, with the same seed, a register gives the
    /// scalar simulator's outcomes and leaves the same state, however the
    /// state is chunked; resets and further gates follow on from it.
    #[test]
    fn measurement_matches_scalar_with_the_same_seed() {
        let _turn = crate::spec::gpu_turn();
        pollster::block_on(async {
            for limit in CHUNK_LIMITS {
                for seed in [1, 7, 42] {
                    let Some(gpu) = gpu_with(10, limit) else {
                        return;
                    };
                    let mut gpu = gpu.unwrap().with_seed(seed);
                    let mut cpu = cpu(10, seed);
                    let what = format!("{} chunks, seed {seed}", gpu.num_chunks());
                    for b in [&mut gpu as &mut dyn QuantumBackend, &mut cpu] {
                        wide_gates(b);
                    }
                    let qubits = [9, 0, 5, 8];
                    let g = gpu.measure_all(&qubits).await.unwrap();
                    let c = cpu.measure_all(&qubits).await.unwrap();
                    assert_eq!(g, c, "{what}: outcomes");
                    assert_same_state(&gpu, &cpu, &what).await;

                    for b in [&mut gpu as &mut dyn QuantumBackend, &mut cpu] {
                        b.reset(3).await.unwrap();
                        b.reset(9).await.unwrap();
                        wide_gates(b);
                    }
                    assert_eq!(gpu.measure(2).await.unwrap(), cpu.measure(2).await.unwrap());
                    assert_same_state(&gpu, &cpu, &what).await;
                }
            }
        })
    }

    /// A collapsed state sums to one and keeps only the measured outcome.
    #[test]
    fn measurement_collapses_and_renormalizes() {
        let _turn = crate::spec::gpu_turn();
        pollster::block_on(async {
            for limit in CHUNK_LIMITS {
                let Some(gpu) = gpu_with(10, limit) else {
                    return;
                };
                let mut gpu = gpu.unwrap().with_seed(3);
                wide_gates(&mut gpu);
                let outcomes = gpu.measure_all(&[1, 9]).await.unwrap();
                let amps = gpu.amplitudes().await.unwrap();
                let norm: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
                assert!((norm - 1.0).abs() < 1e-4, "norm {norm}");
                for (i, a) in amps.iter().enumerate() {
                    let agrees =
                        (i >> 1 & 1 == 1) == outcomes[0] && (i >> 9 & 1 == 1) == outcomes[1];
                    assert!(agrees || a.norm() == 0.0, "|{i}⟩ survived as {a}");
                }
            }
        })
    }

    /// `reset_state` returns every chunk to |0…0⟩ and drops the phase.
    #[test]
    fn reset_state_clears_every_chunk() {
        let _turn = crate::spec::gpu_turn();
        pollster::block_on(async {
            for limit in CHUNK_LIMITS {
                let Some(gpu) = gpu_with(10, limit) else {
                    return;
                };
                let mut gpu = gpu.unwrap();
                wide_gates(&mut gpu);
                gpu.reset_state().unwrap();
                let amps = gpu.amplitudes().await.unwrap();
                assert_eq!(amps[0], Complex::new(1.0, 0.0));
                assert!(amps[1..].iter().all(|a| a.norm() == 0.0));
            }
        })
    }

    /// Shots drawn on the device follow the Born rule, are reproducible
    /// per seed, and on a GHZ state match the scalar simulator draw for
    /// draw.
    #[test]
    fn sampled_shots_follow_the_born_rule() {
        let _turn = crate::spec::gpu_turn();
        pollster::block_on(async {
            for limit in CHUNK_LIMITS {
                let Some(gpu) = gpu_with(10, limit) else {
                    return;
                };
                let mut gpu = gpu.unwrap();
                let mut cpu = cpu(10, 0);
                let plain = GateModifiers::none();
                for b in [&mut gpu as &mut dyn QuantumBackend, &mut cpu] {
                    b.u(0, FRAC_PI_2, 0.0, PI, &plain).unwrap();
                    for q in 1..10 {
                        b.u(q, PI, 0.0, PI, &GateModifiers::ctrl(&[q - 1])).unwrap();
                    }
                }
                let shots = gpu.sample_shots(2000, 11).await.unwrap().unwrap();
                let again = gpu.sample_shots(2000, 11).await.unwrap().unwrap();
                assert_eq!(shots, again, "reproducible per seed");
                let reference = cpu.sample_shots(2000, 11).await.unwrap().unwrap();
                assert_eq!(shots, reference, "{} chunks", gpu.num_chunks());
                let ones = shots.iter().filter(|s| s.iter().all(|&b| b)).count();
                let zeros = shots.iter().filter(|s| s.iter().all(|&b| !b)).count();
                assert_eq!(ones + zeros, 2000);
                assert!((ones as i64 - 1000).abs() < 150, "ones {ones}");

                // A biased chunk-bit qubit: P(q9 = 1) = sin²(0.6) ≈ 0.319.
                gpu.reset_state().unwrap();
                gpu.u(9, 1.2, 0.0, 0.0, &plain).unwrap();
                let shots = gpu.sample_shots(4000, 5).await.unwrap().unwrap();
                let ones = shots.iter().filter(|s| s[9]).count() as f64 / 4000.0;
                assert!((ones - 0.6f64.sin().powi(2)).abs() < 0.03, "P(1) {ones}");
                assert!(shots.iter().all(|s| s[..9].iter().all(|&b| !b)));
            }
        })
    }

    /// A program without qubits still gets a (one-amplitude) state.
    #[test]
    fn zero_qubits_is_a_single_amplitude() {
        let _turn = crate::spec::gpu_turn();
        pollster::block_on(async {
            let Some(gpu) = gpu(0) else { return };
            let mut gpu = gpu.unwrap();
            gpu.reset_state().unwrap();
            assert_eq!(
                gpu.amplitudes().await.unwrap(),
                vec![Complex::new(1.0, 0.0)]
            );
            assert_eq!(
                gpu.sample_shots(3, 1).await.unwrap().unwrap(),
                vec![Vec::<bool>::new(); 3]
            );
        })
    }

    /// A device error reported asynchronously surfaces as a backend error
    /// from the next operation, not as a panic.
    #[test]
    fn a_device_error_surfaces_from_the_next_operation() {
        let _turn = crate::spec::gpu_turn();
        pollster::block_on(async {
            let Some(gpu) = gpu(2) else { return };
            let mut gpu = gpu.unwrap();
            *gpu.device_error.lock().unwrap() = Some("lost".into());
            let err = gpu.u(0, PI, 0.0, PI, &GateModifiers::none()).unwrap_err();
            assert!(matches!(err, ErrorKind::Backend(why) if why.contains("lost")));
            assert!(gpu.u(0, PI, 0.0, PI, &GateModifiers::none()).is_ok());
        })
    }

    /// More than 32 qubits can't be indexed by the kernels: a clean error,
    /// not a host allocation failure or a device validation panic.
    #[test]
    fn oversized_state_is_too_many_qubits() {
        let _turn = crate::spec::gpu_turn();
        pollster::block_on(async {
            for n in [33, 40, 64, 200] {
                let Some(gpu) = gpu(n) else { return };
                assert!(matches!(
                    gpu.err().map(|e| e.kind),
                    Some(ErrorKind::TooManyQubits { requested }) if requested == n
                ));
            }
        })
    }
}
