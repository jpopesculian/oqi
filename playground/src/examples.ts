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

// All three measurements agree; change the seed to
// flip between 000 and 111.
qubit[3] q;
bit[3] c;

h q[0];
cx q[0], q[1];
cx q[1], q[2];

c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
`,
    inputs: '{}',
    seed: '1',
  },
];
