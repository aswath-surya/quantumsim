// lowered for C-3PQ from n08/cal/cal_n08_probe/cal_n08_probe_k000.qasm
// batch=b0003_n08_topk prefix=s00000 group=0
OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
ry(0.1582790499248212) q[0];
ry(0.22407528530181925) q[1];
ry(0.2747243978341321) q[2];
ry(0.31756042929152134) q[3];
ry(0.3554212016902235) q[4];
ry(0.3897607327974748) q[5];
ry(0.4214420015175629) q[6];
ry(0.4510268117962624) q[7];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
