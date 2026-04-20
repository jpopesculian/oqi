defcalgrammar "openpulse";

cal {
  extern port d0;
  extern port d1;
  frame drive0 = newframe(d0, 5.0e9, 0.0);
  frame drive1 = newframe(d1, 5.1e9, 0.0);
  frame cr1 = newframe(d0, 5.2e9, 0.0);
  frame meas0 = newframe(d0, 6.0e9, 0.0);
}

defcal x $0 {
   waveform wf = gaussian(0.1, 160dt, 40dt);
   play(drive0, wf);
}

defcal x $1 {
  waveform wf = gaussian(0.1, 160dt, 40dt);
  play(drive1, wf);
}

defcal rz(angle[20] theta) q {
  shift_phase(drive0, -theta);
}

defcal measure $0 -> bit {
  complex[float] iq;
  bit state;
  waveform wf = gaussian(0.1, 160dt, 40dt);
  play(meas0, wf);
  iq = capture(meas0, 2048);
  return threshold(iq, 1234);
}

defcal zx90_ix $0, $1 {
  waveform wf = gaussian(0.1, 160dt, 40dt);
  play(cr1, wf);
}

defcal cx $0, $1 {
  zx90_ix $0, $1;
  x $0;
  shift_phase(cr1, 0.5);
  zx90_ix $0, $1;
  x $0;
  x $1;
}
