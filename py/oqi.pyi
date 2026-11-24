"""Compile, run and sample OpenQASM 3 programs.

``compile(program)`` turns OpenQASM source into bytecode (or disassembles
bytecode); ``run(program, ...)`` runs a program once on a simulator and
``sample(program, shots=..., ...)`` runs it many times and counts the outputs.
Programs are source text (``str``) or bytecode (``bytes``). Errors raise
``OqiError``.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Literal, TypeAlias

Program: TypeAlias = str | bytes
"""OpenQASM 3 source text, or bytecode from ``compile``."""

InputValue: TypeAlias = bool | int | float | complex | str | Sequence["InputValue"]
"""A value for an ``input`` declaration: a scalar, text such as ``"0101"`` for
``bit[n]``, ``"1.5-2im"`` for ``complex`` or ``"100ns"`` for ``duration``, or
(nested) sequences for arrays."""

Value: TypeAlias = bool | int | float | complex | str | list["Value"]
"""A value as Python sees it: ``bool``, exact ``int``, ``float``, ``complex``,
``str`` (bit registers, angles, durations as OpenQASM text), or ``list``."""

ExternFunction: TypeAlias = Callable[..., Value | None]
"""A synchronous implementation of an ``extern`` function."""

Backend: TypeAlias = Literal[
    "scalar", "rayon", "simd", "rayon-simd", "gpu", "prefer-gpu", "auto", "sum", "mps"
]
"""A simulator, by the ``oqi`` CLI's ``--backend`` names."""

Precision: TypeAlias = Literal["f64", "f32"]
"""Amplitude precision of the CPU state-vector simulators."""

class OqiError(Exception):
    """An error in a program, or in the options it was given.

    The message is the rendered diagnostic.
    """

    code: str | None
    """The diagnostic's stable code (e.g. ``"S0001"`` for a syntax error,
    ``"C0001"`` compile, ``"R0002"`` runtime); None for errors that are not
    program diagnostics, such as invalid options."""

    span: tuple[int, int] | None
    """``(start, end)`` character offsets of the offending text in the
    program's source, or None."""

class CompileResult:
    """The result of ``compile``: the program as bytecode, plus its disassembly."""

    @property
    def bytecode(self) -> bytes:
        """The program as bytecode: ``run`` and ``sample`` take it in place of
        the source."""
    @property
    def disassembly(self) -> str:
        """Textual disassembly of the bytecode."""

class RunResult:
    """The result of ``run``."""

    @property
    def measurements(self) -> list[tuple[int, bool]]:
        """``(qubit, outcome)`` for every measurement, in program order; the
        qubit is its index in the program's qubit memory."""
    @property
    def outputs(self) -> dict[str, Value]:
        """The program's output variables, by name, in program order."""
    @property
    def statevector(self) -> list[complex] | None:
        """The final state vector, when ``statevector=True`` was given."""
    @property
    def backend(self) -> Backend:
        """The backend that ran (``"prefer-gpu"`` resolves to ``"gpu"`` or
        ``"scalar"``)."""
    @property
    def precision(self) -> Precision:
        """The amplitude precision that ran."""

class SampleResult:
    """The result of ``sample``."""

    @property
    def shots(self) -> int:
        """The number of shots that ran."""
    @property
    def histograms(self) -> dict[str, dict[str, int]]:
        """Per output variable (in program order), how often each value was
        observed: ``{name: {label: count}}``, labels sorted by value."""
    @property
    def backend(self) -> Backend:
        """The backend that ran."""
    @property
    def precision(self) -> Precision:
        """The amplitude precision that ran."""

def compile(program: Program) -> CompileResult:
    """Compile an OpenQASM 3 program (source text) to bytecode, or
    disassemble bytecode. Raises ``OqiError`` for an error in the program."""

def run(
    program: Program,
    *,
    inputs: Mapping[str, InputValue] | None = None,
    seed: int | None = None,
    statevector: bool = False,
    externs: Mapping[str, ExternFunction] | None = None,
    timings: Mapping[str, str] | None = None,
    dt: str | None = None,
    backend: Backend = "scalar",
    precision: Precision = "f64",
    max_rank: int | None = None,
    max_bond: int | None = None,
) -> RunResult:
    """Run an OpenQASM 3 program once on a simulator.

    ``timings`` maps operation names (``measure``/``reset`` reserved) to
    duration strings ("50ns", "4dt") and resolves ``durationof`` at compile
    time; ``dt`` sets the device time unit. Both apply to source only.
    ``max_rank`` is the stabilizer-term budget of ``"auto"``/``"sum"`` and
    ``max_bond`` the bond-dimension cap of ``"mps"``.
    """

def sample(
    program: Program,
    *,
    shots: int = 1024,
    inputs: Mapping[str, InputValue] | None = None,
    seed: int | None = None,
    externs: Mapping[str, ExternFunction] | None = None,
    timings: Mapping[str, str] | None = None,
    dt: str | None = None,
    backend: Backend = "scalar",
    precision: Precision = "f64",
    max_rank: int | None = None,
    max_bond: int | None = None,
) -> SampleResult:
    """Run an OpenQASM 3 program for ``shots`` shots and count each output's
    values. A given ``seed`` reproduces the whole histogram."""
