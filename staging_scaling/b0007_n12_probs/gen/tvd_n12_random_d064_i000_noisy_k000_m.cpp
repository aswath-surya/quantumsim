#include "tvd_n12_random_d064_i000_noisy_k000_m.hpp"

#include <mpi.h>

#include "tvd_n12_random_d064_i000_noisy_k000_l1_c0x0.hpp"

void s00001_apply_l0_c0(std::complex<double> *temp0, std::complex<double> *temp1, double *time_result, int runs){
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	std::complex<double> *constant_values = (std::complex<double>*) malloc(1 * sizeof(std::complex<double>));
	constant_values[0] = std::complex<double>(0.9238795325112867, -0.3826834323650898);

	double t0, t1;
	std::complex<double>* constant_values_dev = NULL;

	std::complex<double> *in_dev = (std::complex<double>*) malloc(4096ll * sizeof(std::complex<double>));
	std::complex<double> *out_dev = (std::complex<double>*) malloc(4096ll * sizeof(std::complex<double>));
	memcpy(in_dev, temp0, 4096ll * sizeof(std::complex<double>));
	memcpy(out_dev, temp1, 4096ll * sizeof(std::complex<double>));

	constant_values_dev = (std::complex<double>*) malloc(1ll * sizeof(std::complex<double>));
	memcpy(constant_values_dev, constant_values, 1ll * sizeof(std::complex<double>));

	for(int i = 0; i < runs; ++i) {
		memcpy(in_dev, temp0, 4096ll * sizeof(std::complex<double>));
		memcpy(out_dev, temp1, 4096ll * sizeof(std::complex<double>));

		t0 = MPI_Wtime();
		s00001_apply_l1_c0x0(in_dev, constant_values_dev, rank);
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


