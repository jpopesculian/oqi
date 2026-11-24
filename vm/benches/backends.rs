//! Wall-clock comparison of the CPU simulator backends.
//!
//! Each variant applies the same gate workload to a freshly-allocated state
//! (allocation is excluded from timing) so the numbers reflect gate-kernel
//! throughput, not setup. Run with:
//!
//! ```text
//! cargo bench -p oqi-vm
//! RUSTFLAGS="-C target-cpu=native" cargo bench -p oqi-vm   # enable AVX2
//! ```

use std::f64::consts::{FRAC_PI_2, PI};

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use oqi_vm::{
    DEFAULT_SEED as SEED, GateModifiers, MpsSim, QuantumBackend, SimdSim, StateVectorSim,
};

/// A representative workload: a Hadamard layer, several rotation layers
/// (uncontrolled — the vectorized path), then a CX ladder (controlled — the
/// scalar fallback path).
fn workload(b: &mut dyn QuantumBackend, n: u32, layers: usize) {
    let none = GateModifiers::none();
    for q in 0..n {
        b.u(q, FRAC_PI_2, 0.0, PI, &none).unwrap(); // H
    }
    for _ in 0..layers {
        for q in 0..n {
            b.u(q, 0.1, 0.0, 0.0, &none).unwrap(); // small rotation
        }
    }
    for q in 0..n - 1 {
        b.u(q + 1, PI, 0.0, PI, &GateModifiers::ctrl(&[q])).unwrap(); // CX(q -> q+1)
    }
}

type Maker = Box<dyn Fn() -> Box<dyn QuantumBackend>>;

fn variants(n: u32) -> Vec<(&'static str, Maker)> {
    vec![
        (
            "scalar-f64",
            Box::new(move || {
                Box::new(
                    StateVectorSim::<f64>::new(n)
                        .map(|s| s.with_seed(SEED))
                        .unwrap(),
                )
            }),
        ),
        (
            "rayon-f64",
            Box::new(move || {
                Box::new(
                    StateVectorSim::<f64>::new(n)
                        .map(|s| s.with_seed(SEED))
                        .unwrap()
                        .with_parallel(true),
                )
            }),
        ),
        (
            "simd-f64",
            Box::new(move || Box::new(SimdSim::<f64>::new(n).map(|s| s.with_seed(SEED)).unwrap())),
        ),
        (
            "rayon-simd-f64",
            Box::new(move || {
                Box::new(
                    SimdSim::<f64>::new(n)
                        .map(|s| s.with_seed(SEED))
                        .unwrap()
                        .with_parallel(true),
                )
            }),
        ),
        (
            "mps-f64",
            Box::new(move || Box::new(MpsSim::new(n).unwrap().with_seed(SEED))),
        ),
        (
            "scalar-f32",
            Box::new(move || {
                Box::new(
                    StateVectorSim::<f32>::new(n)
                        .map(|s| s.with_seed(SEED))
                        .unwrap(),
                )
            }),
        ),
        (
            "rayon-f32",
            Box::new(move || {
                Box::new(
                    StateVectorSim::<f32>::new(n)
                        .map(|s| s.with_seed(SEED))
                        .unwrap()
                        .with_parallel(true),
                )
            }),
        ),
        (
            "simd-f32",
            Box::new(move || Box::new(SimdSim::<f32>::new(n).map(|s| s.with_seed(SEED)).unwrap())),
        ),
        (
            "rayon-simd-f32",
            Box::new(move || {
                Box::new(
                    SimdSim::<f32>::new(n)
                        .map(|s| s.with_seed(SEED))
                        .unwrap()
                        .with_parallel(true),
                )
            }),
        ),
    ]
}

fn bench_backends(c: &mut Criterion) {
    let n = 20u32;
    let layers = 8;

    let mut group = c.benchmark_group("apply_workload_20q");
    group.sample_size(10);
    for (name, make) in variants(n) {
        group.bench_function(name, |b| {
            b.iter_batched_ref(
                &make,
                |sim| workload(&mut **sim, n, layers),
                BatchSize::SmallInput,
            );
        });
    }

    // The GPU backend (single precision; needs `--features gpu` and an
    // adapter). The device is created once and reused — recreating it per
    // iteration would dominate. State isn't reset between iterations: gate
    // timing is independent of amplitude values, so reusing the state is
    // fair and avoids a re-upload. Each iteration ends with a readback to
    // force GPU completion, so we time execution rather than just command
    // submission (this adds one readback the CPU variants don't pay).
    #[cfg(feature = "gpu")]
    match pollster::block_on(oqi_vm::GpuSim::new(n)) {
        Ok(mut sim) => {
            group.bench_function("gpu-f32", |b| {
                b.iter(|| {
                    workload(&mut sim, n, layers);
                    let _ = pollster::block_on(sim.amplitudes());
                });
            });
        }
        Err(e) => eprintln!("skipping gpu benchmark: {e}"),
    }

    group.finish();
}

criterion_group!(benches, bench_backends);
criterion_main!(benches);
