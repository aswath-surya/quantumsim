#include <omp.h>

#include "cal_n24_probe_k000_l1_c0x0.hpp"

inline __attribute__((always_inline)) void local_s00000_apply_l2_c0x0x0(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//ry	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8388608ll; local_index += 1){
		std::complex<double> value = constant_values[0];
		size_t idx = 1ll * (0) + 2ll * (local_index % 8388608ll);
		std::complex<double> t0 = temp[idx + 0 * 1];
		std::complex<double> t1 = temp[idx + 1 * 1];
		double t20 = value.real() * t0.real() + value.imag() * t1.real();
		double t21 = value.real() * t0.imag() + value.imag() * t1.imag();
		double t30 = value.real() * t1.real() - value.imag() * t0.real();
		double t31 = value.real() * t1.imag() - value.imag() * t0.imag();
		temp[idx + 0 * 1] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00000_apply_l2_c0x0x1(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//ry	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8388608ll; local_index += 1){
		std::complex<double> value = constant_values[1];
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 4194304ll);
		std::complex<double> t0 = temp[idx + 0 * 2];
		std::complex<double> t1 = temp[idx + 1 * 2];
		double t20 = value.real() * t0.real() + value.imag() * t1.real();
		double t21 = value.real() * t0.imag() + value.imag() * t1.imag();
		double t30 = value.real() * t1.real() - value.imag() * t0.real();
		double t31 = value.real() * t1.imag() - value.imag() * t0.imag();
		temp[idx + 0 * 2] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00000_apply_l2_c0x0x2(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//ry	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8388608ll; local_index += 1){
		std::complex<double> value = constant_values[2];
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 2097152ll);
		std::complex<double> t0 = temp[idx + 0 * 4];
		std::complex<double> t1 = temp[idx + 1 * 4];
		double t20 = value.real() * t0.real() + value.imag() * t1.real();
		double t21 = value.real() * t0.imag() + value.imag() * t1.imag();
		double t30 = value.real() * t1.real() - value.imag() * t0.real();
		double t31 = value.real() * t1.imag() - value.imag() * t0.imag();
		temp[idx + 0 * 4] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00000_apply_l2_c0x0x3(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//ry	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8388608ll; local_index += 1){
		std::complex<double> value = constant_values[3];
		size_t idx = 1ll * (local_index % 8ll) + 16ll * ((local_index / 8ll) % 1048576ll);
		std::complex<double> t0 = temp[idx + 0 * 8];
		std::complex<double> t1 = temp[idx + 1 * 8];
		double t20 = value.real() * t0.real() + value.imag() * t1.real();
		double t21 = value.real() * t0.imag() + value.imag() * t1.imag();
		double t30 = value.real() * t1.real() - value.imag() * t0.real();
		double t31 = value.real() * t1.imag() - value.imag() * t0.imag();
		temp[idx + 0 * 8] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00000_apply_l2_c0x0x4(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//ry	['q4']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8388608ll; local_index += 1){
		std::complex<double> value = constant_values[4];
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 524288ll);
		std::complex<double> t0 = temp[idx + 0 * 16];
		std::complex<double> t1 = temp[idx + 1 * 16];
		double t20 = value.real() * t0.real() + value.imag() * t1.real();
		double t21 = value.real() * t0.imag() + value.imag() * t1.imag();
		double t30 = value.real() * t1.real() - value.imag() * t0.real();
		double t31 = value.real() * t1.imag() - value.imag() * t0.imag();
		temp[idx + 0 * 16] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00000_apply_l2_c0x0x5(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//ry	['q5']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8388608ll; local_index += 1){
		std::complex<double> value = constant_values[5];
		size_t idx = 1ll * (local_index % 32ll) + 64ll * ((local_index / 32ll) % 262144ll);
		std::complex<double> t0 = temp[idx + 0 * 32];
		std::complex<double> t1 = temp[idx + 1 * 32];
		double t20 = value.real() * t0.real() + value.imag() * t1.real();
		double t21 = value.real() * t0.imag() + value.imag() * t1.imag();
		double t30 = value.real() * t1.real() - value.imag() * t0.real();
		double t31 = value.real() * t1.imag() - value.imag() * t0.imag();
		temp[idx + 0 * 32] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00000_apply_l2_c0x0x6(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//ry	['q6']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8388608ll; local_index += 1){
		std::complex<double> value = constant_values[6];
		size_t idx = 1ll * (local_index % 64ll) + 128ll * ((local_index / 64ll) % 131072ll);
		std::complex<double> t0 = temp[idx + 0 * 64];
		std::complex<double> t1 = temp[idx + 1 * 64];
		double t20 = value.real() * t0.real() + value.imag() * t1.real();
		double t21 = value.real() * t0.imag() + value.imag() * t1.imag();
		double t30 = value.real() * t1.real() - value.imag() * t0.real();
		double t31 = value.real() * t1.imag() - value.imag() * t0.imag();
		temp[idx + 0 * 64] = std::complex<double>(t20, t21);
		temp[idx + 1 * 64] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00000_apply_l2_c0x0x7(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//ry	['q7']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8388608ll; local_index += 1){
		std::complex<double> value = constant_values[7];
		size_t idx = 1ll * (local_index % 128ll) + 256ll * ((local_index / 128ll) % 65536ll);
		std::complex<double> t0 = temp[idx + 0 * 128];
		std::complex<double> t1 = temp[idx + 1 * 128];
		double t20 = value.real() * t0.real() + value.imag() * t1.real();
		double t21 = value.real() * t0.imag() + value.imag() * t1.imag();
		double t30 = value.real() * t1.real() - value.imag() * t0.real();
		double t31 = value.real() * t1.imag() - value.imag() * t0.imag();
		temp[idx + 0 * 128] = std::complex<double>(t20, t21);
		temp[idx + 1 * 128] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00000_apply_l2_c0x0x8(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//ry	['q8']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8388608ll; local_index += 1){
		std::complex<double> value = constant_values[8];
		size_t idx = 1ll * (local_index % 256ll) + 512ll * ((local_index / 256ll) % 32768ll);
		std::complex<double> t0 = temp[idx + 0 * 256];
		std::complex<double> t1 = temp[idx + 1 * 256];
		double t20 = value.real() * t0.real() + value.imag() * t1.real();
		double t21 = value.real() * t0.imag() + value.imag() * t1.imag();
		double t30 = value.real() * t1.real() - value.imag() * t0.real();
		double t31 = value.real() * t1.imag() - value.imag() * t0.imag();
		temp[idx + 0 * 256] = std::complex<double>(t20, t21);
		temp[idx + 1 * 256] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00000_apply_l2_c0x0x9(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//ry	['q9']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8388608ll; local_index += 1){
		std::complex<double> value = constant_values[9];
		size_t idx = 1ll * (local_index % 512ll) + 1024ll * ((local_index / 512ll) % 16384ll);
		std::complex<double> t0 = temp[idx + 0 * 512];
		std::complex<double> t1 = temp[idx + 1 * 512];
		double t20 = value.real() * t0.real() + value.imag() * t1.real();
		double t21 = value.real() * t0.imag() + value.imag() * t1.imag();
		double t30 = value.real() * t1.real() - value.imag() * t0.real();
		double t31 = value.real() * t1.imag() - value.imag() * t0.imag();
		temp[idx + 0 * 512] = std::complex<double>(t20, t21);
		temp[idx + 1 * 512] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00000_apply_l2_c0x0x10(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//ry	['q10']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8388608ll; local_index += 1){
		std::complex<double> value = constant_values[10];
		size_t idx = 1ll * (local_index % 1024ll) + 2048ll * ((local_index / 1024ll) % 8192ll);
		std::complex<double> t0 = temp[idx + 0 * 1024];
		std::complex<double> t1 = temp[idx + 1 * 1024];
		double t20 = value.real() * t0.real() + value.imag() * t1.real();
		double t21 = value.real() * t0.imag() + value.imag() * t1.imag();
		double t30 = value.real() * t1.real() - value.imag() * t0.real();
		double t31 = value.real() * t1.imag() - value.imag() * t0.imag();
		temp[idx + 0 * 1024] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1024] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00000_apply_l2_c0x0x11(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//ry	['q11']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8388608ll; local_index += 1){
		std::complex<double> value = constant_values[11];
		size_t idx = 1ll * (local_index % 2048ll) + 4096ll * ((local_index / 2048ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 0 * 2048];
		std::complex<double> t1 = temp[idx + 1 * 2048];
		double t20 = value.real() * t0.real() + value.imag() * t1.real();
		double t21 = value.real() * t0.imag() + value.imag() * t1.imag();
		double t30 = value.real() * t1.real() - value.imag() * t0.real();
		double t31 = value.real() * t1.imag() - value.imag() * t0.imag();
		temp[idx + 0 * 2048] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2048] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00000_apply_l2_c0x0x12(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//ry	['q12']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8388608ll; local_index += 1){
		std::complex<double> value = constant_values[12];
		size_t idx = 1ll * (local_index % 4096ll) + 8192ll * ((local_index / 4096ll) % 2048ll);
		std::complex<double> t0 = temp[idx + 0 * 4096];
		std::complex<double> t1 = temp[idx + 1 * 4096];
		double t20 = value.real() * t0.real() + value.imag() * t1.real();
		double t21 = value.real() * t0.imag() + value.imag() * t1.imag();
		double t30 = value.real() * t1.real() - value.imag() * t0.real();
		double t31 = value.real() * t1.imag() - value.imag() * t0.imag();
		temp[idx + 0 * 4096] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4096] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00000_apply_l2_c0x0x13(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//ry	['q13']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8388608ll; local_index += 1){
		std::complex<double> value = constant_values[13];
		size_t idx = 1ll * (local_index % 8192ll) + 16384ll * ((local_index / 8192ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 0 * 8192];
		std::complex<double> t1 = temp[idx + 1 * 8192];
		double t20 = value.real() * t0.real() + value.imag() * t1.real();
		double t21 = value.real() * t0.imag() + value.imag() * t1.imag();
		double t30 = value.real() * t1.real() - value.imag() * t0.real();
		double t31 = value.real() * t1.imag() - value.imag() * t0.imag();
		temp[idx + 0 * 8192] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8192] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00000_apply_l2_c0x0x14(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//ry	['q14']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8388608ll; local_index += 1){
		std::complex<double> value = constant_values[14];
		size_t idx = 1ll * (local_index % 16384ll) + 32768ll * ((local_index / 16384ll) % 512ll);
		std::complex<double> t0 = temp[idx + 0 * 16384];
		std::complex<double> t1 = temp[idx + 1 * 16384];
		double t20 = value.real() * t0.real() + value.imag() * t1.real();
		double t21 = value.real() * t0.imag() + value.imag() * t1.imag();
		double t30 = value.real() * t1.real() - value.imag() * t0.real();
		double t31 = value.real() * t1.imag() - value.imag() * t0.imag();
		temp[idx + 0 * 16384] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16384] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00000_apply_l2_c0x0x15(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//ry	['q15']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8388608ll; local_index += 1){
		std::complex<double> value = constant_values[15];
		size_t idx = 1ll * (local_index % 32768ll) + 65536ll * ((local_index / 32768ll) % 256ll);
		std::complex<double> t0 = temp[idx + 0 * 32768];
		std::complex<double> t1 = temp[idx + 1 * 32768];
		double t20 = value.real() * t0.real() + value.imag() * t1.real();
		double t21 = value.real() * t0.imag() + value.imag() * t1.imag();
		double t30 = value.real() * t1.real() - value.imag() * t0.real();
		double t31 = value.real() * t1.imag() - value.imag() * t0.imag();
		temp[idx + 0 * 32768] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32768] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00000_apply_l2_c0x0x16(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//ry	['q16']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8388608ll; local_index += 1){
		std::complex<double> value = constant_values[16];
		size_t idx = 1ll * (local_index % 65536ll) + 131072ll * ((local_index / 65536ll) % 128ll);
		std::complex<double> t0 = temp[idx + 0 * 65536];
		std::complex<double> t1 = temp[idx + 1 * 65536];
		double t20 = value.real() * t0.real() + value.imag() * t1.real();
		double t21 = value.real() * t0.imag() + value.imag() * t1.imag();
		double t30 = value.real() * t1.real() - value.imag() * t0.real();
		double t31 = value.real() * t1.imag() - value.imag() * t0.imag();
		temp[idx + 0 * 65536] = std::complex<double>(t20, t21);
		temp[idx + 1 * 65536] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00000_apply_l2_c0x0x17(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//ry	['q17']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8388608ll; local_index += 1){
		std::complex<double> value = constant_values[17];
		size_t idx = 1ll * (local_index % 131072ll) + 262144ll * ((local_index / 131072ll) % 64ll);
		std::complex<double> t0 = temp[idx + 0 * 131072];
		std::complex<double> t1 = temp[idx + 1 * 131072];
		double t20 = value.real() * t0.real() + value.imag() * t1.real();
		double t21 = value.real() * t0.imag() + value.imag() * t1.imag();
		double t30 = value.real() * t1.real() - value.imag() * t0.real();
		double t31 = value.real() * t1.imag() - value.imag() * t0.imag();
		temp[idx + 0 * 131072] = std::complex<double>(t20, t21);
		temp[idx + 1 * 131072] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00000_apply_l2_c0x0x18(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//ry	['q18']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8388608ll; local_index += 1){
		std::complex<double> value = constant_values[18];
		size_t idx = 1ll * (local_index % 262144ll) + 524288ll * ((local_index / 262144ll) % 32ll);
		std::complex<double> t0 = temp[idx + 0 * 262144];
		std::complex<double> t1 = temp[idx + 1 * 262144];
		double t20 = value.real() * t0.real() + value.imag() * t1.real();
		double t21 = value.real() * t0.imag() + value.imag() * t1.imag();
		double t30 = value.real() * t1.real() - value.imag() * t0.real();
		double t31 = value.real() * t1.imag() - value.imag() * t0.imag();
		temp[idx + 0 * 262144] = std::complex<double>(t20, t21);
		temp[idx + 1 * 262144] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00000_apply_l2_c0x0x19(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//ry	['q19']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8388608ll; local_index += 1){
		std::complex<double> value = constant_values[19];
		size_t idx = 1ll * (local_index % 524288ll) + 1048576ll * ((local_index / 524288ll) % 16ll);
		std::complex<double> t0 = temp[idx + 0 * 524288];
		std::complex<double> t1 = temp[idx + 1 * 524288];
		double t20 = value.real() * t0.real() + value.imag() * t1.real();
		double t21 = value.real() * t0.imag() + value.imag() * t1.imag();
		double t30 = value.real() * t1.real() - value.imag() * t0.real();
		double t31 = value.real() * t1.imag() - value.imag() * t0.imag();
		temp[idx + 0 * 524288] = std::complex<double>(t20, t21);
		temp[idx + 1 * 524288] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00000_apply_l2_c0x0x20(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//ry	['q20']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8388608ll; local_index += 1){
		std::complex<double> value = constant_values[20];
		size_t idx = 1ll * (local_index % 1048576ll) + 2097152ll * ((local_index / 1048576ll) % 8ll);
		std::complex<double> t0 = temp[idx + 0 * 1048576];
		std::complex<double> t1 = temp[idx + 1 * 1048576];
		double t20 = value.real() * t0.real() + value.imag() * t1.real();
		double t21 = value.real() * t0.imag() + value.imag() * t1.imag();
		double t30 = value.real() * t1.real() - value.imag() * t0.real();
		double t31 = value.real() * t1.imag() - value.imag() * t0.imag();
		temp[idx + 0 * 1048576] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1048576] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00000_apply_l2_c0x0x21(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//ry	['q21']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8388608ll; local_index += 1){
		std::complex<double> value = constant_values[21];
		size_t idx = 1ll * (local_index % 2097152ll) + 4194304ll * ((local_index / 2097152ll) % 4ll);
		std::complex<double> t0 = temp[idx + 0 * 2097152];
		std::complex<double> t1 = temp[idx + 1 * 2097152];
		double t20 = value.real() * t0.real() + value.imag() * t1.real();
		double t21 = value.real() * t0.imag() + value.imag() * t1.imag();
		double t30 = value.real() * t1.real() - value.imag() * t0.real();
		double t31 = value.real() * t1.imag() - value.imag() * t0.imag();
		temp[idx + 0 * 2097152] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2097152] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00000_apply_l2_c0x0x22(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//ry	['q22']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8388608ll; local_index += 1){
		std::complex<double> value = constant_values[22];
		size_t idx = 1ll * (local_index % 4194304ll) + 8388608ll * ((local_index / 4194304ll) % 2ll);
		std::complex<double> t0 = temp[idx + 0 * 4194304];
		std::complex<double> t1 = temp[idx + 1 * 4194304];
		double t20 = value.real() * t0.real() + value.imag() * t1.real();
		double t21 = value.real() * t0.imag() + value.imag() * t1.imag();
		double t30 = value.real() * t1.real() - value.imag() * t0.real();
		double t31 = value.real() * t1.imag() - value.imag() * t0.imag();
		temp[idx + 0 * 4194304] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4194304] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00000_apply_l2_c0x0x23(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//ry	['q23']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8388608ll; local_index += 1){
		std::complex<double> value = constant_values[23];
		size_t idx = 1ll * (local_index % 8388608ll) + 16777216ll * (0);
		std::complex<double> t0 = temp[idx + 0 * 8388608];
		std::complex<double> t1 = temp[idx + 1 * 8388608];
		double t20 = value.real() * t0.real() + value.imag() * t1.real();
		double t21 = value.real() * t0.imag() + value.imag() * t1.imag();
		double t30 = value.real() * t1.real() - value.imag() * t0.real();
		double t31 = value.real() * t1.imag() - value.imag() * t0.imag();
		temp[idx + 0 * 8388608] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8388608] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void s00000_apply_l2_c0x0x0(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00000_apply_l2_c0x0x0(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00000_apply_l2_c0x0x1(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00000_apply_l2_c0x0x1(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00000_apply_l2_c0x0x2(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00000_apply_l2_c0x0x2(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00000_apply_l2_c0x0x3(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00000_apply_l2_c0x0x3(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00000_apply_l2_c0x0x4(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00000_apply_l2_c0x0x4(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00000_apply_l2_c0x0x5(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00000_apply_l2_c0x0x5(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00000_apply_l2_c0x0x6(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00000_apply_l2_c0x0x6(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00000_apply_l2_c0x0x7(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00000_apply_l2_c0x0x7(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00000_apply_l2_c0x0x8(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00000_apply_l2_c0x0x8(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00000_apply_l2_c0x0x9(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00000_apply_l2_c0x0x9(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00000_apply_l2_c0x0x10(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00000_apply_l2_c0x0x10(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00000_apply_l2_c0x0x11(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00000_apply_l2_c0x0x11(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00000_apply_l2_c0x0x12(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00000_apply_l2_c0x0x12(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00000_apply_l2_c0x0x13(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00000_apply_l2_c0x0x13(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00000_apply_l2_c0x0x14(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00000_apply_l2_c0x0x14(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00000_apply_l2_c0x0x15(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00000_apply_l2_c0x0x15(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00000_apply_l2_c0x0x16(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00000_apply_l2_c0x0x16(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00000_apply_l2_c0x0x17(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00000_apply_l2_c0x0x17(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00000_apply_l2_c0x0x18(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00000_apply_l2_c0x0x18(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00000_apply_l2_c0x0x19(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00000_apply_l2_c0x0x19(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00000_apply_l2_c0x0x20(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00000_apply_l2_c0x0x20(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00000_apply_l2_c0x0x21(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00000_apply_l2_c0x0x21(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00000_apply_l2_c0x0x22(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00000_apply_l2_c0x0x22(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00000_apply_l2_c0x0x23(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00000_apply_l2_c0x0x23(temp, constant_values, iter0);
}

void s00000_apply_l1_c0x0(std::complex<double> *io_temp, std::complex<double>* constant_values, size_t iter0){
	s00000_apply_l2_c0x0x0(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x1(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x2(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x3(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x4(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x5(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x6(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x7(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x8(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x9(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x10(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x11(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x12(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x13(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x14(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x15(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x16(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x17(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x18(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x19(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x20(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x21(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x22(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x23(io_temp, constant_values, iter0);
}

