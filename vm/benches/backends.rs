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
use oqi_vm::{GateModifiers, QuantumBackend, SimdSim, StateVectorSim};

const SEED: u64 = 0x2545F4914F6CDD1D;

/// A representative workload: a Hadamard layer, several rotation layers
/// (uncontrolled — the vectorized path), then a CX ladder (controlled — the
/// scalar fallback path).
fn workload(b: &mut dyn QuantumBackend, n: u32, layers: usize) {
    let none = GateModifiers::none();
    for q in 0..n {
        b.u(q, FRAC_PI_2, 0.0, PI, &none); // H
    }
    for _ in 0..layers {
        for q in 0..n {
            b.u(q, 0.1, 0.0, 0.0, &none); // small rotation
        }
    }
    for q in 0..n - 1 {
        let c = GateModifiers {
            controls: vec![q],
            neg_controls: vec![],
            power: 1.0,
        };
        b.u(q + 1, PI, 0.0, PI, &c); // CX(q -> q+1)
    }
}

type Maker = Box<dyn Fn() -> Box<dyn QuantumBackend>>;

fn variants(n: u32) -> Vec<(&'static str, Maker)> {
    vec![
        (
            "scalar-f64",
            Box::new(move || Box::new(StateVectorSim::<f64>::try_zeroed(n, SEED).unwrap())),
        ),
        (
            "rayon-f64",
            Box::new(move || {
                Box::new(
                    StateVectorSim::<f64>::try_zeroed(n, SEED)
                        .unwrap()
                        .with_parallel(true),
                )
            }),
        ),
        (
            "simd-f64",
            Box::new(move || Box::new(SimdSim::<f64>::try_zeroed(n, SEED).unwrap())),
        ),
        (
            "rayon-simd-f64",
            Box::new(move || {
                Box::new(
                    SimdSim::<f64>::try_zeroed(n, SEED)
                        .unwrap()
                        .with_parallel(true),
                )
            }),
        ),
        (
            "scalar-f32",
            Box::new(move || Box::new(StateVectorSim::<f32>::try_zeroed(n, SEED).unwrap())),
        ),
        (
            "rayon-f32",
            Box::new(move || {
                Box::new(
                    StateVectorSim::<f32>::try_zeroed(n, SEED)
                        .unwrap()
                        .with_parallel(true),
                )
            }),
        ),
        (
            "simd-f32",
            Box::new(move || Box::new(SimdSim::<f32>::try_zeroed(n, SEED).unwrap())),
        ),
        (
            "rayon-simd-f32",
            Box::new(move || {
                Box::new(
                    SimdSim::<f32>::try_zeroed(n, SEED)
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
    group.finish();
}

criterion_group!(benches, bench_backends);
criterion_main!(benches);
