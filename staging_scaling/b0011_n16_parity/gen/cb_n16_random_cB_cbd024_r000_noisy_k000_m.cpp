#include "cb_n16_random_cB_cbd024_r000_noisy_k000_m.hpp"

#include <mpi.h>

#include "cb_n16_random_cB_cbd024_r000_noisy_k000_l1_c0x0.hpp"

void s00003_apply_l0_c0(std::complex<double> *temp0, std::complex<double> *temp1, double *time_result, int runs){
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	double t0, t1;
	std::complex<double>* constant_values_dev = NULL;

	std::complex<double> *in_dev = (std::complex<double>*) malloc(65536ll * sizeof(std::complex<double>));
	std::complex<double> *out_dev = (std::complex<double>*) malloc(65536ll * sizeof(std::complex<double>));
	memcpy(in_dev, temp0, 65536ll * sizeof(std::complex<double>));
	memcpy(out_dev, temp1, 65536ll * sizeof(std::complex<double>));

	for(int i = 0; i < runs; ++i) {
		memcpy(in_dev, temp0, 65536ll * sizeof(std::complex<double>));
		memcpy(out_dev, temp1, 65536ll * sizeof(std::complex<double>));

		t0 = MPI_Wtime();
		s00003_apply_l1_c0x0(in_dev, constant_values_dev, rank);
		t1 = MPI_Wtime();
		*(time_result + 0) += (i / (runs / 2)) * (t1 - t0);
	}

	memcpy(temp0, in_dev, 65536ll * sizeof(std::complex<double>));
	memcpy(temp1, out_dev, 65536ll * sizeof(std::complex<double>));

	free(in_dev);
	free(out_dev);
}


