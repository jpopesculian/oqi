import math

import pytest

import oqi

BELL = """
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c[0] = measure q[0];
c[1] = measure q[1];
"""


def test_compile_bell():
    result = oqi.compile(BELL)
    assert isinstance(result.bytecode, bytes) and result.bytecode
    assert ".module openqasm 3" in result.disassembly
    assert ".proc" in result.disassembly


def test_run_bell_deterministic():
    out = oqi.run(BELL, seed=1234, statevector=True)
    # Bell correlation: both measurements agree, on distinct qubits.
    assert len(out.measurements) == 2
    (q0, v0), (q1, v1) = out.measurements
    assert q0 != q1
    assert v0 == v1
    assert "c" in out.outputs
    # 2 qubits -> 4 amplitudes as native complex numbers.
    assert len(out.statevector) == 4
    assert all(isinstance(a, complex) for a in out.statevector)

    # Fixed seed makes the run deterministic; statevector off by default.
    again = oqi.run(BELL, seed=1234)
    assert again.measurements == out.measurements
    assert again.statevector is None


def test_syntax_error_is_reported_under_the_syntax_code():
    with pytest.raises(oqi.OqiError, match="S0001") as excinfo:
        oqi.compile("OPENQASM 3.0;\nint x = ;\n")
    assert "expected expression" in str(excinfo.value)
    assert "unsupported" not in str(excinfo.value)


def test_include_path_rejected():
    with pytest.raises(oqi.OqiError, match="file includes are not supported"):
        oqi.compile('OPENQASM 3.0;\ninclude "./foo.qasm";\n')


def test_unknown_input_rejected():
    with pytest.raises(oqi.OqiError, match="no input named"):
        oqi.run(BELL, inputs={"nope": 1})


def test_int_input_drives_branch():
    src = (
        'include "stdgates.inc";\n'
        "input int n;\n"
        "qubit q;\n"
        "if (n == 1) { x q; }\n"
        "bit c = measure q;\n"
    )
    assert oqi.run(src, inputs={"n": 1}).measurements == [(0, True)]
    assert oqi.run(src, inputs={"n": 0}).measurements == [(0, False)]


ARRAY_SUM = (
    "OPENQASM 3.0;\n"
    "input array[int, 3] xs;\n"
    "output int total;\n"
    "total = xs[0] + xs[1] + xs[2];\n"
)


def test_array_input():
    assert oqi.run(ARRAY_SUM, inputs={"xs": [1, 2, 3]}).outputs["total"] == 6
    # tuples are accepted too
    assert oqi.run(ARRAY_SUM, inputs={"xs": (4, 5, 6)}).outputs["total"] == 15


def test_array_input_wrong_length_rejected():
    with pytest.raises(oqi.OqiError, match=r"input .xs."):
        oqi.run(ARRAY_SUM, inputs={"xs": [1, 2]})


INC = (
    "OPENQASM 3.0;\n"
    "qubit q;\n"
    "extern inc(int[32]) -> int[32];\n"
    "int[32] x = 41;\n"
    "int[32] y = inc(x);\n"
)

LOG_IT = "OPENQASM 3.0;\nqubit q;\nextern log_it(int[32]);\nlog_it(7);\n"


def test_extern_callback():
    out = oqi.run(INC, externs={"inc": lambda x: x + 1})
    assert out.outputs["y"] == 42


def test_extern_void_return_ignored():
    seen = []
    oqi.run(LOG_IT, externs={"log_it": lambda x: seen.append(x) or 123})
    assert seen == [7]


def test_extern_raising_callback():
    def boom(x):
        raise ValueError("boom")

    with pytest.raises(oqi.OqiError, match="extern function `log_it` failed"):
        oqi.run(LOG_IT, externs={"log_it": boom})
    with pytest.raises(oqi.OqiError, match="boom"):
        oqi.run(LOG_IT, externs={"log_it": boom})


def test_extern_missing_rejected():
    with pytest.raises(oqi.OqiError, match="extern function `inc` is not provided"):
        oqi.run(INC)


def test_extern_angle_return():
    src = (
        "OPENQASM 3.0;\n"
        "qubit q;\n"
        "extern get_theta() -> angle[16];\n"
        "angle[16] a = get_theta();\n"
    )
    out = oqi.run(src, externs={"get_theta": lambda: math.pi / 2})
    assert out.outputs["a"] == "(π/2)"


def test_extern_bitreg_round_trip():
    src = (
        "OPENQASM 3.0;\n"
        "qubit q;\n"
        "extern flip(bit[4]) -> bit[4];\n"
        'bit[4] r = "0011";\n'
        "bit[4] s = flip(r);\n"
    )
    seen = []

    def flip(bits):
        seen.append(bits)
        return bits[::-1]

    out = oqi.run(src, externs={"flip": flip})
    # Args arrive as unquoted MSB-first strings; outputs keep the quoted
    # OpenQASM text form.
    assert seen == ["0011"]
    assert out.outputs["s"] == '"1100"'


def test_extern_bad_return_value():
    with pytest.raises(oqi.OqiError, match="extern function `inc` failed"):
        oqi.run(INC, externs={"inc": lambda x: 1.5})


def test_extern_non_callable_rejected():
    with pytest.raises(oqi.OqiError, match="not callable"):
        oqi.run(INC, externs={"inc": 5})


def test_extern_unused_is_allowed():
    oqi.run(BELL, seed=1, externs={"unused": lambda: 0})


def test_extern_coroutine_rejected():
    async def inc(x):
        return x + 1

    with pytest.raises(oqi.OqiError, match="must be synchronous"):
        oqi.run(INC, externs={"inc": inc})


TIMED = (
    'include "stdgates.inc";\n'
    "qubit q;\n"
    "duration d = durationof({x q; delay[30ns] q;});\n"
    "bit c = measure q;\n"
)


def test_run_with_timings_resolves_durationof():
    # With a timing table, `durationof` resolves at compile time.
    out = oqi.run(TIMED, timings={"x": "20ns"})
    assert out.outputs["d"] == "50ns"

    # dt-valued timings resolve against the `dt` option.
    out = oqi.run(TIMED, timings={"x": "40dt"}, dt="0.5ns")
    assert out.outputs["d"] == "50ns"

    # Without timings, the VM's runtime path still answers (x is 0ns).
    out = oqi.run(TIMED)
    assert out.outputs["d"] == "30ns"


def test_bad_timing_literal_raises():
    with pytest.raises(oqi.OqiError, match="is not a duration literal"):
        oqi.run(TIMED, timings={"x": "abc"})


def test_angle_and_bitreg_inputs():
    src = (
        "OPENQASM 3.0;\n"
        "input angle[32] theta;\n"
        "input bit[4] mask;\n"
        "output bit[4] copy;\n"
        "output float[64] sine;\n"
        "copy = mask;\n"
        "sine = sin(theta);\n"
    )
    out = oqi.run(src, inputs={"theta": 0.5, "mask": "0110"})
    assert out.outputs["copy"] == '"0110"'
    assert out.outputs["sine"] == pytest.approx(math.sin(0.5), abs=1e-6)

    with pytest.raises(oqi.OqiError, match="4-character string of 0s and 1s"):
        oqi.run(src, inputs={"theta": 0.5, "mask": "01"})
    with pytest.raises(oqi.OqiError, match="unsupported Python type `dict`"):
        oqi.run(src, inputs={"theta": {}, "mask": "0110"})


@pytest.mark.parametrize(
    "backend", ["scalar", "rayon", "simd", "rayon-simd", "auto", "sum", "mps"]
)
def test_backends_share_the_cli_names(backend):
    out = oqi.run(BELL, seed=7, backend=backend)
    assert out.backend == backend
    (_, v0), (_, v1) = out.measurements
    assert v0 == v1


def test_backend_options():
    assert oqi.run(BELL).precision == "f64"
    assert oqi.run(BELL, backend="simd", precision="f32").precision == "f32"
    # The auto-router is exact (f64) whatever was asked for.
    assert oqi.run(BELL, backend="auto", precision="f32").precision == "f64"
    oqi.run(BELL, backend="sum", max_rank=4)
    assert oqi.run(BELL, backend="mps", precision="f32", max_bond=4).precision == "f64"

    with pytest.raises(oqi.OqiError, match="unknown backend `cpu`"):
        oqi.run(BELL, backend="cpu")
    with pytest.raises(oqi.OqiError, match="unknown precision"):
        oqi.run(BELL, precision="f16")
    # The wheel is built without the wgpu simulator.
    with pytest.raises(oqi.OqiError, match="backend unavailable"):
        oqi.run(BELL, backend="gpu")
    assert oqi.run(BELL, backend="prefer-gpu").backend == "scalar"


def test_sample_bell_histogram():
    out = oqi.sample(BELL, shots=200, seed=1234)
    assert out.shots == 200
    assert out.backend == "scalar"
    counts = out.histograms["c"]
    # A Bell state only ever yields the correlated outcomes, sorted by value.
    assert list(counts) == ["00", "11"]
    assert sum(counts.values()) == 200

    # Same seed, same histogram; 1024 shots by default.
    assert oqi.sample(BELL, shots=200, seed=1234).histograms == out.histograms
    assert sum(oqi.sample(BELL).histograms["c"].values()) == 1024

    with pytest.raises(oqi.OqiError, match="at least 1"):
        oqi.sample(BELL, shots=0)


@pytest.mark.parametrize("backend", ["scalar", "simd", "auto"])
def test_sample_with_feedback(backend):
    # A gate conditioned on a measurement can't be sampled from one final
    # state, so every shot re-runs the circuit: `b` mirrors `a` each time.
    src = (
        'include "stdgates.inc";\n'
        "input int flips;\n"
        "qubit[2] q;\n"
        "h q[0];\n"
        "bit a = measure q[0];\n"
        "if (a) { x q[1]; }\n"
        "for int i in [1:flips] { x q[1]; }\n"
        "bit b = measure q[1];\n"
    )
    out = oqi.sample(src, shots=128, seed=5, inputs={"flips": 2}, backend=backend)
    a, b = out.histograms["a"], out.histograms["b"]
    assert sum(a.values()) == sum(b.values()) == 128
    assert set(a) == {"0", "1"}
    assert a == b
