#include "cal_n04_probe_k000_m.hpp"

#include <mpi.h>

#include "cal_n04_probe_k000_l1_c0x0.hpp"

void s00000_apply_l0_c0(std::complex<double> *temp0, std::complex<double> *temp1, double *time_result, int runs){
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	std::complex<double> *constant_values = (std::complex<double>*) malloc(4 * sizeof(std::complex<double>));
	constant_values[0] = std::complex<double>(0.9874208829065749, -0.15811388300841897);
	constant_values[1] = std::complex<double>(0.9746794344808964, -0.22360679774997896);
	constant_values[2] = std::complex<double>(0.9617692030835673, -0.27386127875258304);
	constant_values[3] = std::complex<double>(0.9486832980505138, -0.31622776601683794);

	double t0, t1;
	std::complex<double>* constant_values_dev = NULL;

	std::complex<double> *in_dev = (std::complex<double>*) malloc(16ll * sizeof(std::complex<double>));
	std::complex<double> *out_dev = (std::complex<double>*) malloc(16ll * sizeof(std::complex<double>));
	memcpy(in_dev, temp0, 16ll * sizeof(std::complex<double>));
	memcpy(out_dev, temp1, 16ll * sizeof(std::complex<double>));

	constant_values_dev = (std::complex<double>*) malloc(4ll * sizeof(std::complex<double>));
	memcpy(constant_values_dev, constant_values, 4ll * sizeof(std::complex<double>));

	for(int i = 0; i < runs; ++i) {
		memcpy(in_dev, temp0, 16ll * sizeof(std::complex<double>));
		memcpy(out_dev, temp1, 16ll * sizeof(std::complex<double>));

		t0 = MPI_Wtime();
		s00000_apply_l1_c0x0(in_dev, constant_values_dev, rank);
		t1 = MPI_Wtime();
		*(time_result + 0) += (i / (runs / 2)) * (t1 - t0);
	}

	free(constant_values);

	memcpy(temp0, in_dev, 16ll * sizeof(std::complex<double>));
	memcpy(temp1, out_dev, 16ll * sizeof(std::complex<double>));

	free(in_dev);
	free(out_dev);
	free(constant_values_dev);
}


