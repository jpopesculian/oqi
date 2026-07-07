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
