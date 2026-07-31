#include "cal_n20_probe_k000_m.hpp"

#include <mpi.h>

#include "cal_n20_probe_k000_l1_c0x0.hpp"

void s00000_apply_l0_c0(std::complex<double> *temp0, std::complex<double> *temp1, double *time_result, int runs){
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	std::complex<double> *constant_values = (std::complex<double>*) malloc(20 * sizeof(std::complex<double>));
	constant_values[0] = std::complex<double>(0.999499874937461, -0.03162277660168379);
	constant_values[1] = std::complex<double>(0.9989994994993742, -0.044721359549995794);
	constant_values[2] = std::complex<double>(0.9984988733093293, -0.05477225575051661);
	constant_values[3] = std::complex<double>(0.997997995989972, -0.06324555320336758);
	constant_values[4] = std::complex<double>(0.9974968671630001, -0.07071067811865475);
	constant_values[5] = std::complex<double>(0.9969954864491614, -0.07745966692414834);
	constant_values[6] = std::complex<double>(0.9964938534682489, -0.08366600265340755);
	constant_values[7] = std::complex<double>(0.9959919678390986, -0.08944271909999159);
	constant_values[8] = std::complex<double>(0.9954898291795853, -0.09486832980505139);
	constant_values[9] = std::complex<double>(0.99498743710662, -0.1);
	constant_values[10] = std::complex<double>(0.9944847912361455, -0.10488088481701516);
	constant_values[11] = std::complex<double>(0.9939818911831342, -0.10954451150103323);
	constant_values[12] = std::complex<double>(0.9934787365615834, -0.1140175425099138);
	constant_values[13] = std::complex<double>(0.9929753269845127, -0.11832159566199232);
	constant_values[14] = std::complex<double>(0.9924716620639604, -0.1224744871391589);
	constant_values[15] = std::complex<double>(0.9919677414109795, -0.12649110640673517);
	constant_values[16] = std::complex<double>(0.991463564635635, -0.130384048104053);
	constant_values[17] = std::complex<double>(0.9909591313469996, -0.1341640786499874);
	constant_values[18] = std::complex<double>(0.9904544411531506, -0.13784048752090222);
	constant_values[19] = std::complex<double>(0.9899494936611666, -0.1414213562373095);

	double t0, t1;
	std::complex<double>* constant_values_dev = NULL;

	std::complex<double> *in_dev = (std::complex<double>*) malloc(1048576ll * sizeof(std::complex<double>));
	std::complex<double> *out_dev = (std::complex<double>*) malloc(1048576ll * sizeof(std::complex<double>));
	memcpy(in_dev, temp0, 1048576ll * sizeof(std::complex<double>));
	memcpy(out_dev, temp1, 1048576ll * sizeof(std::complex<double>));

	constant_values_dev = (std::complex<double>*) malloc(20ll * sizeof(std::complex<double>));
	memcpy(constant_values_dev, constant_values, 20ll * sizeof(std::complex<double>));

	for(int i = 0; i < runs; ++i) {
		memcpy(in_dev, temp0, 1048576ll * sizeof(std::complex<double>));
		memcpy(out_dev, temp1, 1048576ll * sizeof(std::complex<double>));

		t0 = MPI_Wtime();
		s00000_apply_l1_c0x0(in_dev, constant_values_dev, rank);
		t1 = MPI_Wtime();
		*(time_result + 0) += (i / (runs / 2)) * (t1 - t0);
	}

	free(constant_values);

	memcpy(temp0, in_dev, 1048576ll * sizeof(std::complex<double>));
	memcpy(temp1, out_dev, 1048576ll * sizeof(std::complex<double>));

	free(in_dev);
	free(out_dev);
	free(constant_values_dev);
}


