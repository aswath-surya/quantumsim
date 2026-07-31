#include "cal_n24_probe_k000_m.hpp"

#include <mpi.h>

#include "cal_n24_probe_k000_l1_c0x0.hpp"

void s00000_apply_l0_c0(std::complex<double> *temp0, std::complex<double> *temp1, double *time_result, int runs){
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	std::complex<double> *constant_values = (std::complex<double>*) malloc(24 * sizeof(std::complex<double>));
	constant_values[0] = std::complex<double>(0.9996527174752018, -0.026352313834736494);
	constant_values[1] = std::complex<double>(0.9993053142614179, -0.037267799624996496);
	constant_values[2] = std::complex<double>(0.9989577902327338, -0.04564354645876384);
	constant_values[3] = std::complex<double>(0.9986101452630162, -0.05270462766947299);
	constant_values[4] = std::complex<double>(0.9982623792259117, -0.05892556509887896);
	constant_values[5] = std::complex<double>(0.9979144919948469, -0.06454972243679027);
	constant_values[6] = std::complex<double>(0.9975664834430279, -0.06972166887783963);
	constant_values[7] = std::complex<double>(0.9972183534434395, -0.07453559924999299);
	constant_values[8] = std::complex<double>(0.9968701018688443, -0.07905694150420949);
	constant_values[9] = std::complex<double>(0.9965217285917831, -0.08333333333333334);
	constant_values[10] = std::complex<double>(0.9961732334845738, -0.08740073734751264);
	constant_values[11] = std::complex<double>(0.9958246164193104, -0.09128709291752768);
	constant_values[12] = std::complex<double>(0.9954758772678634, -0.0950146187582615);
	constant_values[13] = std::complex<double>(0.9951270159018786, -0.09860132971832694);
	constant_values[14] = std::complex<double>(0.9947780321927768, -0.10206207261596575);
	constant_values[15] = std::complex<double>(0.9944289260117531, -0.10540925533894598);
	constant_values[16] = std::complex<double>(0.9940796972297766, -0.10865337342004415);
	constant_values[17] = std::complex<double>(0.9937303457175896, -0.11180339887498948);
	constant_values[18] = std::complex<double>(0.9933808713457067, -0.11486707293408518);
	constant_values[19] = std::complex<double>(0.9930312739844154, -0.11785113019775792);
	constant_values[20] = std::complex<double>(0.9926815535037743, -0.12076147288491199);
	constant_values[21] = std::complex<double>(0.9923317097736131, -0.12360330811826105);
	constant_values[22] = std::complex<double>(0.991981742663532, -0.12638125740085918);
	constant_values[23] = std::complex<double>(0.9916316520429012, -0.12909944487358055);

	double t0, t1;
	std::complex<double>* constant_values_dev = NULL;

	std::complex<double> *in_dev = (std::complex<double>*) malloc(16777216ll * sizeof(std::complex<double>));
	std::complex<double> *out_dev = (std::complex<double>*) malloc(16777216ll * sizeof(std::complex<double>));
	memcpy(in_dev, temp0, 16777216ll * sizeof(std::complex<double>));
	memcpy(out_dev, temp1, 16777216ll * sizeof(std::complex<double>));

	constant_values_dev = (std::complex<double>*) malloc(24ll * sizeof(std::complex<double>));
	memcpy(constant_values_dev, constant_values, 24ll * sizeof(std::complex<double>));

	for(int i = 0; i < runs; ++i) {
		memcpy(in_dev, temp0, 16777216ll * sizeof(std::complex<double>));
		memcpy(out_dev, temp1, 16777216ll * sizeof(std::complex<double>));

		t0 = MPI_Wtime();
		s00000_apply_l1_c0x0(in_dev, constant_values_dev, rank);
		t1 = MPI_Wtime();
		*(time_result + 0) += (i / (runs / 2)) * (t1 - t0);
	}

	free(constant_values);

	memcpy(temp0, in_dev, 16777216ll * sizeof(std::complex<double>));
	memcpy(temp1, out_dev, 16777216ll * sizeof(std::complex<double>));

	free(in_dev);
	free(out_dev);
	free(constant_values_dev);
}


