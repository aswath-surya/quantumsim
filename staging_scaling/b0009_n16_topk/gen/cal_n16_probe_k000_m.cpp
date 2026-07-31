#include "cal_n16_probe_k000_m.hpp"

#include <mpi.h>

#include "cal_n16_probe_k000_l1_c0x0.hpp"

void s00000_apply_l0_c0(std::complex<double> *temp0, std::complex<double> *temp1, double *time_result, int runs){
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	std::complex<double> *constant_values = (std::complex<double>*) malloc(16 * sizeof(std::complex<double>));
	constant_values[0] = std::complex<double>(0.9992184445855671, -0.03952847075210474);
	constant_values[1] = std::complex<double>(0.9984362773857929, -0.05590169943749474);
	constant_values[2] = std::complex<double>(0.9976534969617458, -0.06846531968814577);
	constant_values[3] = std::complex<double>(0.9968701018688443, -0.07905694150420949);
	constant_values[4] = std::complex<double>(0.9960860906568267, -0.08838834764831845);
	constant_values[5] = std::complex<double>(0.9953014618697191, -0.09682458365518543);
	constant_values[6] = std::complex<double>(0.9945162140458043, -0.10458250331675945);
	constant_values[7] = std::complex<double>(0.9937303457175896, -0.11180339887498948);
	constant_values[8] = std::complex<double>(0.992943855411775, -0.11858541225631422);
	constant_values[9] = std::complex<double>(0.9921567416492215, -0.125);
	constant_values[10] = std::complex<double>(0.9913690029449176, -0.13110110602126895);
	constant_values[11] = std::complex<double>(0.9905806378079475, -0.13693063937629155);
	constant_values[12] = std::complex<double>(0.9897916447414576, -0.14252192813739226);
	constant_values[13] = std::complex<double>(0.9890020222426241, -0.1479019945774904);
	constant_values[14] = std::complex<double>(0.9882117688026185, -0.15309310892394862);
	constant_values[15] = std::complex<double>(0.9874208829065749, -0.15811388300841897);

	double t0, t1;
	std::complex<double>* constant_values_dev = NULL;

	std::complex<double> *in_dev = (std::complex<double>*) malloc(65536ll * sizeof(std::complex<double>));
	std::complex<double> *out_dev = (std::complex<double>*) malloc(65536ll * sizeof(std::complex<double>));
	memcpy(in_dev, temp0, 65536ll * sizeof(std::complex<double>));
	memcpy(out_dev, temp1, 65536ll * sizeof(std::complex<double>));

	constant_values_dev = (std::complex<double>*) malloc(16ll * sizeof(std::complex<double>));
	memcpy(constant_values_dev, constant_values, 16ll * sizeof(std::complex<double>));

	for(int i = 0; i < runs; ++i) {
		memcpy(in_dev, temp0, 65536ll * sizeof(std::complex<double>));
		memcpy(out_dev, temp1, 65536ll * sizeof(std::complex<double>));

		t0 = MPI_Wtime();
		s00000_apply_l1_c0x0(in_dev, constant_values_dev, rank);
		t1 = MPI_Wtime();
		*(time_result + 0) += (i / (runs / 2)) * (t1 - t0);
	}

	free(constant_values);

	memcpy(temp0, in_dev, 65536ll * sizeof(std::complex<double>));
	memcpy(temp1, out_dev, 65536ll * sizeof(std::complex<double>));

	free(in_dev);
	free(out_dev);
	free(constant_values_dev);
}


