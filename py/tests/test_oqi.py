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
