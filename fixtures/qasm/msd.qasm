/*
 * Magic state distillation (serialized form).
 *
 * A serialized rewrite of the two-level 10:2 distillation from
 * arXiv:1811.00566. Instead of running the two second-level distillation
 * chains in parallel on separate registers, they run one after another on the
 * same `magic` and `level1` registers. That holds the simultaneous qubit count
 * at 23 (vs 44 for the parallel form) so the state vector fits in memory,
 * though the full circuit is still expensive to simulate.
 */
include "stdgates.inc";
// Y-basis measurement
def ymeasure(qubit q) -> bit {
  s q;
  h q;
  return measure q;
}

/*
 * Distillation subroutine takes 10 |H> magic states
 * and 3 scratch qubits that will be reinitialized.
 * The first two input magic states are the outputs.
 * The subroutine returns a success bit that is true
 * on success and false otherwise (see arXiv:1811.00566).
 */
def distill(qubit[10] magic, qubit[3] scratch) -> bool {
  bit temp;
  bit[3] checks;
  // Encode two magic states in the [[4,2,2]] code
  reset scratch[0: 1];
  h scratch[1];
  cx scratch[1], magic[0];
  cx magic[1], scratch[0];
  cx magic[0], scratch[0];
  cx scratch[1], magic[1];
  // Body of distillation circuit
  cy magic[2], scratch[0];
  h magic[1];
  temp = ymeasure(magic[2]);
  if(temp == 1) { ry(-pi / 2) scratch[0]; }
  reset scratch[2];
  h scratch[2];
  cz scratch[2], scratch[0];
  cy magic[3], scratch[0];
  temp = ymeasure(magic[3]);
  if(temp==0) { ry(pi / 2) scratch[0]; }
  h scratch[0];
  s scratch[0];
  cy magic[4], scratch[1];
  temp = ymeasure(magic[4]);
  if(temp==1) { ry(-pi / 2) scratch[1]; }
  cz scratch[1], scratch[2];
  cy magic[5], scratch[1];
  temp = ymeasure(magic[5]);
  if(temp==0) { ry(pi / 2) scratch[1]; }
  cy scratch[0], magic[1];
  inv @ s scratch[1];
  cz scratch[0], scratch[1];
  h scratch[0];
  cy scratch[1], magic[1];
  cy magic[6], scratch[0];
  temp = ymeasure(magic[6]);
  if(temp == 1) { ry(-pi / 2) scratch[0]; }
  cz scratch[2], scratch[1];
  cz scratch[2], scratch[0];
  cy magic[7], scratch[0];
  temp = ymeasure(magic[7]);
  if(temp == 0) ry(pi / 2) scratch[0];
  cy magic[8], scratch[1];
  temp = ymeasure(magic[8]);
  if(temp==1) { ry(-pi / 2) scratch[1]; }
  cz scratch[2], scratch[1];
  cy magic[9], scratch[1];
  temp = ymeasure(magic[9]);
  if(temp == 0) { ry(pi / 2) scratch[1]; }
  h scratch[2];
  // Decode [[4,2,2]] code
  cx magic[0], scratch[0];
  cx scratch[1], magic[1];
  cx magic[1], scratch[0];
  cx scratch[1], magic[0];
  h scratch[1];
  checks = measure scratch;
  bool success = !(bool(checks[0]) || bool(checks[1]) || bool(checks[2]));
  return success;
}

// Repeat level-0 distillation until success
def rus_level_0(qubit[10] magic, qubit[3] scratch) {
  bool success;
  while(!success) {
    reset magic;
    ry(pi / 4) magic;
    success = distill(magic, scratch);
  }
}

qubit[10] magic;     // level-0 working register, reused every round
qubit[10] level1;    // accumulates level-0 outputs, then the level-1 input
qubit[3] scratch;    // distillation scratch, reused
bit[2] c;            // computation results

reset magic;
reset level1;
reset scratch;

// Run the two second-level chains serially on the shared registers.
for uint chain in [0:1] {
  // Fill the level-1 input register with ten level-0 outputs.
  for uint i in [0:9] {
    rus_level_0 magic, scratch;
    swap magic[0], level1[i];
  }
  // Second-level distillation.
  bool ok = distill(level1, scratch);

  // Consume one distilled |H> state to apply a T gate to a computation qubit
  // (reusing magic[chain], free now that the level-0 work is done).
  reset magic[chain];
  h magic[chain];
  if (ok) {
    cy level1[0], magic[chain];
    bit outcome = ymeasure(level1[0]);
    if (outcome == 1) ry(pi / 2) magic[chain];
  }
  c[chain] = measure magic[chain];
}
