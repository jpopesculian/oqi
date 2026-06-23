/*
 * Prepare a parameterized number of Bell pairs and teleport a qubit
 * through the chain of them.
 */
include "stdgates.inc";

const int[32] n_pairs = 10;

def bellprep(qubit[2] q) {
  reset q;
  h q[0];
  cx q[0], q[1];
}

def xprepare(qubit q) {
  reset q;
  h q;
}

def teleport(qubit src, qubit[2] bp) {
  bit[2] pf;
  bellprep bp;
  cx src, bp[0];
  h src;
  pf[0] = measure src;
  pf[1] = measure bp[0];
  if (pf[0] == 1) z bp[1];
  if (pf[1] == 1) x bp[1];
}

qubit input_qubit;
bit output_qubit;
qubit[2*n_pairs] q;

xprepare(input_qubit);
rz(pi / 4) input_qubit;

teleport input_qubit, q[0:1];
for uint i in [1: n_pairs - 1] {
  teleport q[2*i - 1], q[{2*i, 2*i + 1}];
}

// Measuring the final target in the X basis reproduces input_qubit's prepared
// distribution (P(0) = cos^2(pi/8) ~= 0.85), confirming the teleport chain.
h q[2*n_pairs - 1];
output_qubit = measure q[2*n_pairs - 1];
