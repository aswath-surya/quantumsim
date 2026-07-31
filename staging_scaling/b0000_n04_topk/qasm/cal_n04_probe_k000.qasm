// lowered for C-3PQ from n04/cal/cal_n04_probe/cal_n04_probe_k000.qasm
// batch=b0000_n04_topk prefix=s00000 group=0
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
ry(0.31756042929152134) q[0];
ry(0.4510268117962624) q[1];
ry(0.5548110329800715) q[2];
ry(0.6435011087932844) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
