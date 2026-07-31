#include "cal_n08_probe_k000_m.hpp"

#include <mpi.h>

#include "cal_n08_probe_k000_l1_c0x0.hpp"

void s00000_apply_l0_c0(std::complex<double> *temp0, std::complex<double> *temp1, double *time_result, int runs){
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	std::complex<double> *constant_values = (std::complex<double>*) malloc(8 * sizeof(std::complex<double>));
	constant_values[0] = std::complex<double>(0.9968701018688443, -0.07905694150420949);
	constant_values[1] = std::complex<double>(0.9937303457175896, -0.11180339887498948);
	constant_values[2] = std::complex<double>(0.9905806378079475, -0.13693063937629155);
	constant_values[3] = std::complex<double>(0.9874208829065749, -0.15811388300841897);
	constant_values[4] = std::complex<double>(0.9842509842514764, -0.1767766952966369);
	constant_values[5] = std::complex<double>(0.9810708435174291, -0.19364916731037085);
	constant_values[6] = std::complex<double>(0.9778803607803973, -0.2091650066335189);
	constant_values[7] = std::complex<double>(0.9746794344808964, -0.22360679774997896);

	double t0, t1;
	std::complex<double>* constant_values_dev = NULL;

	std::complex<double> *in_dev = (std::complex<double>*) malloc(256ll * sizeof(std::complex<double>));
	std::complex<double> *out_dev = (std::complex<double>*) malloc(256ll * sizeof(std::complex<double>));
	memcpy(in_dev, temp0, 256ll * sizeof(std::complex<double>));
	memcpy(out_dev, temp1, 256ll * sizeof(std::complex<double>));

	constant_values_dev = (std::complex<double>*) malloc(8ll * sizeof(std::complex<double>));
	memcpy(constant_values_dev, constant_values, 8ll * sizeof(std::complex<double>));

	for(int i = 0; i < runs; ++i) {
		memcpy(in_dev, temp0, 256ll * sizeof(std::complex<double>));
		memcpy(out_dev, temp1, 256ll * sizeof(std::complex<double>));

		t0 = MPI_Wtime();
		s00000_apply_l1_c0x0(in_dev, constant_values_dev, rank);
		t1 = MPI_Wtime();
		*(time_result + 0) += (i / (runs / 2)) * (t1 - t0);
	}

	free(constant_values);

	memcpy(temp0, in_dev, 256ll * sizeof(std::complex<double>));
	memcpy(temp1, out_dev, 256ll * sizeof(std::complex<double>));

	free(in_dev);
	free(out_dev);
	free(constant_values_dev);
}


