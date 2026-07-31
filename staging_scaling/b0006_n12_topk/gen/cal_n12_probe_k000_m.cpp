#include "cal_n12_probe_k000_m.hpp"

#include <mpi.h>

#include "cal_n12_probe_k000_l1_c0x0.hpp"

void s00000_apply_l0_c0(std::complex<double> *temp0, std::complex<double> *temp1, double *time_result, int runs){
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	std::complex<double> *constant_values = (std::complex<double>*) malloc(12 * sizeof(std::complex<double>));
	constant_values[0] = std::complex<double>(0.9986101452630162, -0.05270462766947299);
	constant_values[1] = std::complex<double>(0.9972183534434395, -0.07453559924999299);
	constant_values[2] = std::complex<double>(0.9958246164193104, -0.09128709291752768);
	constant_values[3] = std::complex<double>(0.9944289260117531, -0.10540925533894598);
	constant_values[4] = std::complex<double>(0.9930312739844154, -0.11785113019775792);
	constant_values[5] = std::complex<double>(0.9916316520429012, -0.12909944487358055);
	constant_values[6] = std::complex<double>(0.9902300518341965, -0.13944333775567927);
	constant_values[7] = std::complex<double>(0.9888264649460884, -0.14907119849998599);
	constant_values[8] = std::complex<double>(0.9874208829065749, -0.15811388300841897);
	constant_values[9] = std::complex<double>(0.9860132971832694, -0.16666666666666669);
	constant_values[10] = std::complex<double>(0.9846036991827953, -0.17480147469502527);
	constant_values[11] = std::complex<double>(0.983192080250175, -0.18257418583505536);

	double t0, t1;
	std::complex<double>* constant_values_dev = NULL;

	std::complex<double> *in_dev = (std::complex<double>*) malloc(4096ll * sizeof(std::complex<double>));
	std::complex<double> *out_dev = (std::complex<double>*) malloc(4096ll * sizeof(std::complex<double>));
	memcpy(in_dev, temp0, 4096ll * sizeof(std::complex<double>));
	memcpy(out_dev, temp1, 4096ll * sizeof(std::complex<double>));

	constant_values_dev = (std::complex<double>*) malloc(12ll * sizeof(std::complex<double>));
	memcpy(constant_values_dev, constant_values, 12ll * sizeof(std::complex<double>));

	for(int i = 0; i < runs; ++i) {
		memcpy(in_dev, temp0, 4096ll * sizeof(std::complex<double>));
		memcpy(out_dev, temp1, 4096ll * sizeof(std::complex<double>));

		t0 = MPI_Wtime();
		s00000_apply_l1_c0x0(in_dev, constant_values_dev, rank);
		t1 = MPI_Wtime();
		*(time_result + 0) += (i / (runs / 2)) * (t1 - t0);
	}

	free(constant_values);

	memcpy(temp0, in_dev, 4096ll * sizeof(std::complex<double>));
	memcpy(temp1, out_dev, 4096ll * sizeof(std::complex<double>));

	free(in_dev);
	free(out_dev);
	free(constant_values_dev);
}


