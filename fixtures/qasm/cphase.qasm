// Build a controlled-NOT from the built-in U gate and the `ctrl` modifier,
// then define a controlled-phase gate from CX and single-qubit rotations.
gate CX a, b { ctrl @ U(π, 0, π) a, b; }

gate cphase(θ) a, b {
  U(0, 0, θ / 2) a;
  CX a, b;
  U(0, 0, -θ / 2) b;
  CX a, b;
  U(0, 0, θ / 2) b;
}

qubit[2] q;
cphase(π / 2) q[0], q[1];
bit[2] c = measure q;
