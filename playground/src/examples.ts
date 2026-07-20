export interface Example {
  name: string;
  source: string;
  /** Pretty-printed JSON for the inputs pane. */
  inputs: string;
  seed: string;
}

export const EXAMPLES: Example[] = [
  {
    name: 'Bell pair',
    source: `OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;
bit[2] c;

h q[0];
cx q[0], q[1];

c[0] = measure q[0];
c[1] = measure q[1];
`,
    inputs: '{}',
    seed: '1234',
  },
  {
    name: 'Parameterized rotation',
    source: `OPENQASM 3.0;
include "stdgates.inc";

// Rotate by an angle supplied at run time.
input float[64] theta;

qubit q;
rx(theta) q;
bit c = measure q;
`,
    inputs: '{\n  "theta": 0.7853981633974483\n}',
    seed: '42',
  },
  {
    name: 'Parity of n',
    source: `OPENQASM 3.0;
include "stdgates.inc";

// Apply X n times: the measured bit is n mod 2.
input int n;

qubit q;
for int i in [1:n] {
  x q;
}
bit parity = measure q;
`,
    inputs: '{\n  "n": 3\n}',
    seed: '7',
  },
  {
    name: 'GHZ state',
    source: `OPENQASM 3.0;
include "stdgates.inc";

// Build a GHZ state whose size is chosen at run time (max 8):
//   0 -> empty, 1 -> H on one qubit, 2 -> Bell pair,
//   3+ -> an n-qubit GHZ chain.
input int size;
qubit[8] q;
bit[8] c;

if (size > 0) {
  h q[0];
}
for int i in [1:size - 1] {
  cx q[i - 1], q[i];
}
c = measure q;
`,
    inputs: '{\n  "size": 3\n}',
    seed: '1',
  },
];
