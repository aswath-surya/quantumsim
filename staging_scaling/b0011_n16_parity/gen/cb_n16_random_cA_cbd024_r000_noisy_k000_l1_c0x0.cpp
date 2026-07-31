#include <omp.h>

#include "cb_n16_random_cA_cbd024_r000_noisy_k000_l1_c0x0.hpp"

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x0(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (0) + 2ll * (local_index % 32768ll);
		std::complex<double> t0 = temp[idx + 0 * 1];
		std::complex<double> t1 = temp[idx + 1 * 1];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 1] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x1(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 8ll) + 16ll * ((local_index / 8ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 0 * 8];
		std::complex<double> t1 = temp[idx + 1 * 8];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 8] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x2(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q4']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 2048ll);
		std::complex<double> t0 = temp[idx + 0 * 16];
		std::complex<double> t1 = temp[idx + 1 * 16];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 16] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x3(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q5']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 32ll) + 64ll * ((local_index / 32ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 0 * 32];
		std::complex<double> t1 = temp[idx + 1 * 32];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 32] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x4(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q9']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 512ll) + 1024ll * ((local_index / 512ll) % 64ll);
		std::complex<double> t0 = temp[idx + 0 * 512];
		std::complex<double> t1 = temp[idx + 1 * 512];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 512] = std::complex<double>(t20, t21);
		temp[idx + 1 * 512] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x5(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q13']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 8192ll) + 16384ll * ((local_index / 8192ll) % 4ll);
		std::complex<double> t0 = temp[idx + 0 * 8192];
		std::complex<double> t1 = temp[idx + 1 * 8192];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 8192] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8192] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x6(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 16384ll);
		std::complex<double> t0 = temp[idx + 0 * 2];
		std::complex<double> t1 = temp[idx + 1 * 2];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 2] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x7(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 8192ll);
		std::complex<double> t0 = temp[idx + 0 * 4];
		std::complex<double> t1 = temp[idx + 1 * 4];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 4] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x8(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q6']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 64ll) + 128ll * ((local_index / 64ll) % 512ll);
		std::complex<double> t0 = temp[idx + 0 * 64];
		std::complex<double> t1 = temp[idx + 1 * 64];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 64] = std::complex<double>(t20, t21);
		temp[idx + 1 * 64] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x9(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q7']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 128ll) + 256ll * ((local_index / 128ll) % 256ll);
		std::complex<double> t0 = temp[idx + 0 * 128];
		std::complex<double> t1 = temp[idx + 1 * 128];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 128] = std::complex<double>(t20, t21);
		temp[idx + 1 * 128] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x10(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q8']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 256ll) + 512ll * ((local_index / 256ll) % 128ll);
		std::complex<double> t0 = temp[idx + 0 * 256];
		std::complex<double> t1 = temp[idx + 1 * 256];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 256] = std::complex<double>(t20, t21);
		temp[idx + 1 * 256] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x11(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q10']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 1024ll) + 2048ll * ((local_index / 1024ll) % 32ll);
		std::complex<double> t0 = temp[idx + 0 * 1024];
		std::complex<double> t1 = temp[idx + 1 * 1024];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 1024] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1024] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x12(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q11']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 2048ll) + 4096ll * ((local_index / 2048ll) % 16ll);
		std::complex<double> t0 = temp[idx + 0 * 2048];
		std::complex<double> t1 = temp[idx + 1 * 2048];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 2048] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2048] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x13(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q12']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 4096ll) + 8192ll * ((local_index / 4096ll) % 8ll);
		std::complex<double> t0 = temp[idx + 0 * 4096];
		std::complex<double> t1 = temp[idx + 1 * 4096];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 4096] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4096] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x14(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q14']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 16384ll) + 32768ll * ((local_index / 16384ll) % 2ll);
		std::complex<double> t0 = temp[idx + 0 * 16384];
		std::complex<double> t1 = temp[idx + 1 * 16384];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 16384] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16384] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x15(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q15']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 32768ll) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 0 * 32768];
		std::complex<double> t1 = temp[idx + 1 * 32768];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 32768] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32768] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x16(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q4']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 2048ll);
		std::complex<double> t0 = temp[idx + 0 * 16];
		std::complex<double> t1 = temp[idx + 1 * 16];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 16] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x17(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (0) + 2ll * (local_index % 32768ll);
		std::complex<double> t0 = temp[idx + 0 * 1];
		std::complex<double> t1 = temp[idx + 1 * 1];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 1] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x18(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 8ll) + 16ll * ((local_index / 8ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 0 * 8];
		std::complex<double> t1 = temp[idx + 1 * 8];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 8] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x19(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q5']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 32ll) + 64ll * ((local_index / 32ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 0 * 32];
		std::complex<double> t1 = temp[idx + 1 * 32];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 32] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x20(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q9']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 512ll) + 1024ll * ((local_index / 512ll) % 64ll);
		std::complex<double> t0 = temp[idx + 0 * 512];
		std::complex<double> t1 = temp[idx + 1 * 512];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 512] = std::complex<double>(t20, t21);
		temp[idx + 1 * 512] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x21(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q13']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 8192ll) + 16384ll * ((local_index / 8192ll) % 4ll);
		std::complex<double> t0 = temp[idx + 0 * 8192];
		std::complex<double> t1 = temp[idx + 1 * 8192];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 8192] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8192] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x22(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q7']	['q6']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 64ll) + 64 + 128ll * (0) + 256ll * ((local_index / 64ll) % 256ll);
		std::complex<double> t0 = temp[idx + 128];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 128] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x23(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q11']	['q10']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 1024ll) + 1024 + 2048ll * (0) + 4096ll * ((local_index / 1024ll) % 16ll);
		std::complex<double> t0 = temp[idx + 2048];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2048] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x24(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q15']	['q14']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 16384 + 32768ll * (0) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 32768];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32768] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x25(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q1']	['q0']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (0) + 1 + 2ll * (0) + 4ll * (local_index % 16384ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x26(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q3']	['q2']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 4 + 8ll * (0) + 16ll * ((local_index / 4ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x27(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q9']	['q8']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 256 + 512ll * (0) + 1024ll * ((local_index / 256ll) % 64ll);
		std::complex<double> t0 = temp[idx + 512];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 512] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x28(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q13']	['q12']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4096ll) + 4096 + 8192ll * (0) + 16384ll * ((local_index / 4096ll) % 4ll);
		std::complex<double> t0 = temp[idx + 8192];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8192] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x29(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q4']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 2048ll);
		std::complex<double> t0 = temp[idx + 0 * 16];
		std::complex<double> t1 = temp[idx + 1 * 16];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 16] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x30(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q6']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 64ll) + 128ll * ((local_index / 64ll) % 512ll);
		std::complex<double> t0 = temp[idx + 0 * 64];
		std::complex<double> t1 = temp[idx + 1 * 64];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 64] = std::complex<double>(t20, t21);
		temp[idx + 1 * 64] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x31(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q7']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 128ll) + 256ll * ((local_index / 128ll) % 256ll);
		std::complex<double> t0 = temp[idx + 0 * 128];
		std::complex<double> t1 = temp[idx + 1 * 128];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 128] = std::complex<double>(t20, t21);
		temp[idx + 1 * 128] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x32(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q10']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 1024ll) + 2048ll * ((local_index / 1024ll) % 32ll);
		std::complex<double> t0 = temp[idx + 0 * 1024];
		std::complex<double> t1 = temp[idx + 1 * 1024];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 1024] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1024] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x33(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q11']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 2048ll) + 4096ll * ((local_index / 2048ll) % 16ll);
		std::complex<double> t0 = temp[idx + 0 * 2048];
		std::complex<double> t1 = temp[idx + 1 * 2048];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 2048] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2048] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x34(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q14']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 16384ll) + 32768ll * ((local_index / 16384ll) % 2ll);
		std::complex<double> t0 = temp[idx + 0 * 16384];
		std::complex<double> t1 = temp[idx + 1 * 16384];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 16384] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16384] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x35(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q15']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 32768ll) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 0 * 32768];
		std::complex<double> t1 = temp[idx + 1 * 32768];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 32768] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32768] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x36(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q5']	['q4']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 16 + 32ll * (0) + 64ll * ((local_index / 16ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x37(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (0) + 2ll * (local_index % 32768ll);
		std::complex<double> t0 = temp[idx + 0 * 1];
		std::complex<double> t1 = temp[idx + 1 * 1];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 1] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x38(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 16384ll);
		std::complex<double> t0 = temp[idx + 0 * 2];
		std::complex<double> t1 = temp[idx + 1 * 2];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 2] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x39(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 8192ll);
		std::complex<double> t0 = temp[idx + 0 * 4];
		std::complex<double> t1 = temp[idx + 1 * 4];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 4] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x40(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * ((local_index / 8ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 0 * 8];
		std::complex<double> t1 = temp[idx + 1 * 8];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 8] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x41(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q8']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 256ll) + 512ll * ((local_index / 256ll) % 128ll);
		std::complex<double> t0 = temp[idx + 0 * 256];
		std::complex<double> t1 = temp[idx + 1 * 256];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 256] = std::complex<double>(t20, t21);
		temp[idx + 1 * 256] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x42(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q9']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 512ll) + 1024ll * ((local_index / 512ll) % 64ll);
		std::complex<double> t0 = temp[idx + 0 * 512];
		std::complex<double> t1 = temp[idx + 1 * 512];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 512] = std::complex<double>(t20, t21);
		temp[idx + 1 * 512] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x43(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q12']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 4096ll) + 8192ll * ((local_index / 4096ll) % 8ll);
		std::complex<double> t0 = temp[idx + 0 * 4096];
		std::complex<double> t1 = temp[idx + 1 * 4096];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 4096] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4096] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x44(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q13']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 8192ll) + 16384ll * ((local_index / 8192ll) % 4ll);
		std::complex<double> t0 = temp[idx + 0 * 8192];
		std::complex<double> t1 = temp[idx + 1 * 8192];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 8192] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8192] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x45(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q4']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 2048ll);
		std::complex<double> t0 = temp[idx + 0 * 16];
		std::complex<double> t1 = temp[idx + 1 * 16];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 16] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x46(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q5']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 32ll) + 64ll * ((local_index / 32ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x47(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q1']	['q0']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (0) + 1 + 2ll * (0) + 4ll * (local_index % 16384ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x48(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q3']	['q2']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 4 + 8ll * (0) + 16ll * ((local_index / 4ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x49(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q9']	['q8']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 256 + 512ll * (0) + 1024ll * ((local_index / 256ll) % 64ll);
		std::complex<double> t0 = temp[idx + 512];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 512] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x50(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q13']	['q12']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4096ll) + 4096 + 8192ll * (0) + 16384ll * ((local_index / 4096ll) % 4ll);
		std::complex<double> t0 = temp[idx + 8192];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8192] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x51(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q7']	['q6']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 64ll) + 64 + 128ll * (0) + 256ll * ((local_index / 64ll) % 256ll);
		std::complex<double> t0 = temp[idx + 128];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 128] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x52(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q11']	['q10']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 1024ll) + 1024 + 2048ll * (0) + 4096ll * ((local_index / 1024ll) % 16ll);
		std::complex<double> t0 = temp[idx + 2048];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2048] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x53(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q15']	['q14']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 16384 + 32768ll * (0) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 32768];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32768] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x54(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q6']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 64ll) + 128ll * ((local_index / 64ll) % 512ll);
		std::complex<double> t0 = temp[idx + 0 * 64];
		std::complex<double> t1 = temp[idx + 1 * 64];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 64] = std::complex<double>(t20, t21);
		temp[idx + 1 * 64] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x55(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q7']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 128ll) + 256ll * ((local_index / 128ll) % 256ll);
		std::complex<double> t0 = temp[idx + 0 * 128];
		std::complex<double> t1 = temp[idx + 1 * 128];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 128] = std::complex<double>(t20, t21);
		temp[idx + 1 * 128] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x56(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q10']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 1024ll) + 2048ll * ((local_index / 1024ll) % 32ll);
		std::complex<double> t0 = temp[idx + 0 * 1024];
		std::complex<double> t1 = temp[idx + 1 * 1024];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 1024] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1024] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x57(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q11']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 2048ll) + 4096ll * ((local_index / 2048ll) % 16ll);
		std::complex<double> t0 = temp[idx + 0 * 2048];
		std::complex<double> t1 = temp[idx + 1 * 2048];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 2048] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2048] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x58(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q14']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 16384ll) + 32768ll * ((local_index / 16384ll) % 2ll);
		std::complex<double> t0 = temp[idx + 0 * 16384];
		std::complex<double> t1 = temp[idx + 1 * 16384];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 16384] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16384] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x59(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q15']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 32768ll) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 32768];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32768] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x60(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q5']	['q4']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 16 + 32ll * (0) + 64ll * ((local_index / 16ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x61(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (0) + 2ll * (local_index % 32768ll);
		std::complex<double> t0 = temp[idx + 0 * 1];
		std::complex<double> t1 = temp[idx + 1 * 1];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 1] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x62(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 16384ll);
		std::complex<double> t0 = temp[idx + 0 * 2];
		std::complex<double> t1 = temp[idx + 1 * 2];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 2] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x63(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 8192ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x64(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * ((local_index / 8ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 0 * 8];
		std::complex<double> t1 = temp[idx + 1 * 8];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 8] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x65(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q8']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 256ll) + 512ll * ((local_index / 256ll) % 128ll);
		std::complex<double> t0 = temp[idx + 0 * 256];
		std::complex<double> t1 = temp[idx + 1 * 256];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 256] = std::complex<double>(t20, t21);
		temp[idx + 1 * 256] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x66(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q9']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 512ll) + 1024ll * ((local_index / 512ll) % 64ll);
		std::complex<double> t0 = temp[idx + 0 * 512];
		std::complex<double> t1 = temp[idx + 1 * 512];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 512] = std::complex<double>(t20, t21);
		temp[idx + 1 * 512] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x67(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q12']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 4096ll) + 8192ll * ((local_index / 4096ll) % 8ll);
		std::complex<double> t0 = temp[idx + 0 * 4096];
		std::complex<double> t1 = temp[idx + 1 * 4096];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 4096] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4096] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x68(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q13']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 8192ll) + 16384ll * ((local_index / 8192ll) % 4ll);
		std::complex<double> t0 = temp[idx + 0 * 8192];
		std::complex<double> t1 = temp[idx + 1 * 8192];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 8192] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8192] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x69(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q7']	['q6']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 64ll) + 64 + 128ll * (0) + 256ll * ((local_index / 64ll) % 256ll);
		std::complex<double> t0 = temp[idx + 128];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 128] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x70(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q11']	['q10']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 1024ll) + 1024 + 2048ll * (0) + 4096ll * ((local_index / 1024ll) % 16ll);
		std::complex<double> t0 = temp[idx + 2048];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2048] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x71(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q15']	['q14']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 16384 + 32768ll * (0) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 32768];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32768] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x72(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q4']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 2048ll);
		std::complex<double> t0 = temp[idx + 0 * 16];
		std::complex<double> t1 = temp[idx + 1 * 16];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 16] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x73(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q5']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 32ll) + 64ll * ((local_index / 32ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 0 * 32];
		std::complex<double> t1 = temp[idx + 1 * 32];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 32] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x74(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q1']	['q0']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (0) + 1 + 2ll * (0) + 4ll * (local_index % 16384ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x75(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q3']	['q2']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 4 + 8ll * (0) + 16ll * ((local_index / 4ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x76(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q9']	['q8']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 256 + 512ll * (0) + 1024ll * ((local_index / 256ll) % 64ll);
		std::complex<double> t0 = temp[idx + 512];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 512] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x77(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q13']	['q12']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4096ll) + 4096 + 8192ll * (0) + 16384ll * ((local_index / 4096ll) % 4ll);
		std::complex<double> t0 = temp[idx + 8192];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8192] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x78(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q10']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 1024ll) + 2048ll * ((local_index / 1024ll) % 32ll);
		std::complex<double> t0 = temp[idx + 0 * 1024];
		std::complex<double> t1 = temp[idx + 1 * 1024];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 1024] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1024] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x79(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q11']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 2048ll) + 4096ll * ((local_index / 2048ll) % 16ll);
		std::complex<double> t0 = temp[idx + 0 * 2048];
		std::complex<double> t1 = temp[idx + 1 * 2048];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 2048] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2048] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x80(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q14']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 16384ll) + 32768ll * ((local_index / 16384ll) % 2ll);
		std::complex<double> t0 = temp[idx + 0 * 16384];
		std::complex<double> t1 = temp[idx + 1 * 16384];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 16384] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16384] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x81(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q15']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 32768ll) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 0 * 32768];
		std::complex<double> t1 = temp[idx + 1 * 32768];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 32768] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32768] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x82(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q5']	['q4']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 16 + 32ll * (0) + 64ll * ((local_index / 16ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x83(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (0) + 2ll * (local_index % 32768ll);
		std::complex<double> t0 = temp[idx + 0 * 1];
		std::complex<double> t1 = temp[idx + 1 * 1];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 1] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x84(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 16384ll);
		std::complex<double> t0 = temp[idx + 0 * 2];
		std::complex<double> t1 = temp[idx + 1 * 2];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 2] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x85(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 8192ll);
		std::complex<double> t0 = temp[idx + 0 * 4];
		std::complex<double> t1 = temp[idx + 1 * 4];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 4] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x86(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * ((local_index / 8ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 0 * 8];
		std::complex<double> t1 = temp[idx + 1 * 8];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 8] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x87(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q8']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 256ll) + 512ll * ((local_index / 256ll) % 128ll);
		std::complex<double> t0 = temp[idx + 0 * 256];
		std::complex<double> t1 = temp[idx + 1 * 256];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 256] = std::complex<double>(t20, t21);
		temp[idx + 1 * 256] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x88(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q9']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 512ll) + 1024ll * ((local_index / 512ll) % 64ll);
		std::complex<double> t0 = temp[idx + 0 * 512];
		std::complex<double> t1 = temp[idx + 1 * 512];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 512] = std::complex<double>(t20, t21);
		temp[idx + 1 * 512] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x89(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q12']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 4096ll) + 8192ll * ((local_index / 4096ll) % 8ll);
		std::complex<double> t0 = temp[idx + 0 * 4096];
		std::complex<double> t1 = temp[idx + 1 * 4096];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 4096] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4096] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x90(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q13']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 8192ll) + 16384ll * ((local_index / 8192ll) % 4ll);
		std::complex<double> t0 = temp[idx + 0 * 8192];
		std::complex<double> t1 = temp[idx + 1 * 8192];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 8192] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8192] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x91(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q6']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 64ll) + 128ll * ((local_index / 64ll) % 512ll);
		std::complex<double> t0 = temp[idx + 0 * 64];
		std::complex<double> t1 = temp[idx + 1 * 64];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 64] = std::complex<double>(t20, t21);
		temp[idx + 1 * 64] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x92(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q7']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 128ll) + 256ll * ((local_index / 128ll) % 256ll);
		std::complex<double> t0 = temp[idx + 128];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 128] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x93(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q7']	['q6']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 64ll) + 64 + 128ll * (0) + 256ll * ((local_index / 64ll) % 256ll);
		std::complex<double> t0 = temp[idx + 128];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 128] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x94(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q11']	['q10']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 1024ll) + 1024 + 2048ll * (0) + 4096ll * ((local_index / 1024ll) % 16ll);
		std::complex<double> t0 = temp[idx + 2048];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2048] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x95(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q15']	['q14']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 16384 + 32768ll * (0) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 32768];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32768] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x96(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q4']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 2048ll);
		std::complex<double> t0 = temp[idx + 0 * 16];
		std::complex<double> t1 = temp[idx + 1 * 16];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 16] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x97(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q5']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 32ll) + 64ll * ((local_index / 32ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 0 * 32];
		std::complex<double> t1 = temp[idx + 1 * 32];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 32] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x98(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q1']	['q0']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (0) + 1 + 2ll * (0) + 4ll * (local_index % 16384ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x99(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q3']	['q2']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 4 + 8ll * (0) + 16ll * ((local_index / 4ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x100(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q9']	['q8']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 256 + 512ll * (0) + 1024ll * ((local_index / 256ll) % 64ll);
		std::complex<double> t0 = temp[idx + 512];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 512] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x101(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q13']	['q12']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4096ll) + 4096 + 8192ll * (0) + 16384ll * ((local_index / 4096ll) % 4ll);
		std::complex<double> t0 = temp[idx + 8192];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8192] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x102(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * ((local_index / 8ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 0 * 8];
		std::complex<double> t1 = temp[idx + 1 * 8];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 8] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x103(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q8']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 256ll) + 512ll * ((local_index / 256ll) % 128ll);
		std::complex<double> t0 = temp[idx + 0 * 256];
		std::complex<double> t1 = temp[idx + 1 * 256];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 256] = std::complex<double>(t20, t21);
		temp[idx + 1 * 256] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x104(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q9']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 512ll) + 1024ll * ((local_index / 512ll) % 64ll);
		std::complex<double> t0 = temp[idx + 0 * 512];
		std::complex<double> t1 = temp[idx + 1 * 512];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 512] = std::complex<double>(t20, t21);
		temp[idx + 1 * 512] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x105(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q12']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 4096ll) + 8192ll * ((local_index / 4096ll) % 8ll);
		std::complex<double> t0 = temp[idx + 0 * 4096];
		std::complex<double> t1 = temp[idx + 1 * 4096];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 4096] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4096] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x106(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q13']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 8192ll) + 16384ll * ((local_index / 8192ll) % 4ll);
		std::complex<double> t0 = temp[idx + 0 * 8192];
		std::complex<double> t1 = temp[idx + 1 * 8192];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 8192] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8192] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x107(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q6']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 64ll) + 128ll * ((local_index / 64ll) % 512ll);
		std::complex<double> t0 = temp[idx + 0 * 64];
		std::complex<double> t1 = temp[idx + 1 * 64];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 64] = std::complex<double>(t20, t21);
		temp[idx + 1 * 64] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x108(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q7']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 128ll) + 256ll * ((local_index / 128ll) % 256ll);
		std::complex<double> t0 = temp[idx + 0 * 128];
		std::complex<double> t1 = temp[idx + 1 * 128];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 128] = std::complex<double>(t20, t21);
		temp[idx + 1 * 128] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x109(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q10']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 1024ll) + 2048ll * ((local_index / 1024ll) % 32ll);
		std::complex<double> t0 = temp[idx + 0 * 1024];
		std::complex<double> t1 = temp[idx + 1 * 1024];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 1024] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1024] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x110(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q11']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 2048ll) + 4096ll * ((local_index / 2048ll) % 16ll);
		std::complex<double> t0 = temp[idx + 0 * 2048];
		std::complex<double> t1 = temp[idx + 1 * 2048];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 2048] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2048] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x111(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q14']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 16384ll) + 32768ll * ((local_index / 16384ll) % 2ll);
		std::complex<double> t0 = temp[idx + 0 * 16384];
		std::complex<double> t1 = temp[idx + 1 * 16384];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 16384] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16384] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x112(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q15']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 32768ll) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 0 * 32768];
		std::complex<double> t1 = temp[idx + 1 * 32768];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 32768] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32768] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x113(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q5']	['q4']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 16 + 32ll * (0) + 64ll * ((local_index / 16ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x114(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (0) + 2ll * (local_index % 32768ll);
		std::complex<double> t0 = temp[idx + 0 * 1];
		std::complex<double> t1 = temp[idx + 1 * 1];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 1] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x115(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 16384ll);
		std::complex<double> t0 = temp[idx + 0 * 2];
		std::complex<double> t1 = temp[idx + 1 * 2];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 2] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x116(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 8192ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x117(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q7']	['q6']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 64ll) + 64 + 128ll * (0) + 256ll * ((local_index / 64ll) % 256ll);
		std::complex<double> t0 = temp[idx + 128];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 128] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x118(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q11']	['q10']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 1024ll) + 1024 + 2048ll * (0) + 4096ll * ((local_index / 1024ll) % 16ll);
		std::complex<double> t0 = temp[idx + 2048];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2048] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x119(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q15']	['q14']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 16384 + 32768ll * (0) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 32768];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32768] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x120(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q4']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 2048ll);
		std::complex<double> t0 = temp[idx + 0 * 16];
		std::complex<double> t1 = temp[idx + 1 * 16];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 16] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x121(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q5']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 32ll) + 64ll * ((local_index / 32ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 0 * 32];
		std::complex<double> t1 = temp[idx + 1 * 32];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 32] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x122(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q1']	['q0']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (0) + 1 + 2ll * (0) + 4ll * (local_index % 16384ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x123(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q3']	['q2']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 4 + 8ll * (0) + 16ll * ((local_index / 4ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x124(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q9']	['q8']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 256 + 512ll * (0) + 1024ll * ((local_index / 256ll) % 64ll);
		std::complex<double> t0 = temp[idx + 512];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 512] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x125(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q13']	['q12']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4096ll) + 4096 + 8192ll * (0) + 16384ll * ((local_index / 4096ll) % 4ll);
		std::complex<double> t0 = temp[idx + 8192];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8192] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x126(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q6']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 64ll) + 128ll * ((local_index / 64ll) % 512ll);
		std::complex<double> t0 = temp[idx + 0 * 64];
		std::complex<double> t1 = temp[idx + 1 * 64];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 64] = std::complex<double>(t20, t21);
		temp[idx + 1 * 64] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x127(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q7']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 128ll) + 256ll * ((local_index / 128ll) % 256ll);
		std::complex<double> t0 = temp[idx + 0 * 128];
		std::complex<double> t1 = temp[idx + 1 * 128];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 128] = std::complex<double>(t20, t21);
		temp[idx + 1 * 128] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x128(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q10']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 1024ll) + 2048ll * ((local_index / 1024ll) % 32ll);
		std::complex<double> t0 = temp[idx + 0 * 1024];
		std::complex<double> t1 = temp[idx + 1 * 1024];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 1024] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1024] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x129(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q11']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 2048ll) + 4096ll * ((local_index / 2048ll) % 16ll);
		std::complex<double> t0 = temp[idx + 0 * 2048];
		std::complex<double> t1 = temp[idx + 1 * 2048];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 2048] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2048] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x130(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q14']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 16384ll) + 32768ll * ((local_index / 16384ll) % 2ll);
		std::complex<double> t0 = temp[idx + 0 * 16384];
		std::complex<double> t1 = temp[idx + 1 * 16384];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 16384] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16384] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x131(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q15']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 32768ll) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 0 * 32768];
		std::complex<double> t1 = temp[idx + 1 * 32768];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 32768] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32768] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x132(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q5']	['q4']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 16 + 32ll * (0) + 64ll * ((local_index / 16ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x133(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (0) + 2ll * (local_index % 32768ll);
		std::complex<double> t0 = temp[idx + 0 * 1];
		std::complex<double> t1 = temp[idx + 1 * 1];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 1] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x134(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 16384ll);
		std::complex<double> t0 = temp[idx + 0 * 2];
		std::complex<double> t1 = temp[idx + 1 * 2];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 2] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x135(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 8192ll);
		std::complex<double> t0 = temp[idx + 0 * 4];
		std::complex<double> t1 = temp[idx + 1 * 4];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 4] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x136(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * ((local_index / 8ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 0 * 8];
		std::complex<double> t1 = temp[idx + 1 * 8];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 8] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x137(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q8']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 256ll) + 512ll * ((local_index / 256ll) % 128ll);
		std::complex<double> t0 = temp[idx + 0 * 256];
		std::complex<double> t1 = temp[idx + 1 * 256];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 256] = std::complex<double>(t20, t21);
		temp[idx + 1 * 256] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x138(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q9']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 512ll) + 1024ll * ((local_index / 512ll) % 64ll);
		std::complex<double> t0 = temp[idx + 0 * 512];
		std::complex<double> t1 = temp[idx + 1 * 512];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 512] = std::complex<double>(t20, t21);
		temp[idx + 1 * 512] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x139(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q12']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 4096ll) + 8192ll * ((local_index / 4096ll) % 8ll);
		std::complex<double> t0 = temp[idx + 0 * 4096];
		std::complex<double> t1 = temp[idx + 1 * 4096];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 4096] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4096] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x140(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q13']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 8192ll) + 16384ll * ((local_index / 8192ll) % 4ll);
		std::complex<double> t0 = temp[idx + 0 * 8192];
		std::complex<double> t1 = temp[idx + 1 * 8192];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 8192] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8192] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x141(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q4']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 2048ll);
		std::complex<double> t0 = temp[idx + 0 * 16];
		std::complex<double> t1 = temp[idx + 1 * 16];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 16] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x142(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q5']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 32ll) + 64ll * ((local_index / 32ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 0 * 32];
		std::complex<double> t1 = temp[idx + 1 * 32];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 32] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x143(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q1']	['q0']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (0) + 1 + 2ll * (0) + 4ll * (local_index % 16384ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x144(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q3']	['q2']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 4 + 8ll * (0) + 16ll * ((local_index / 4ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x145(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q9']	['q8']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 256 + 512ll * (0) + 1024ll * ((local_index / 256ll) % 64ll);
		std::complex<double> t0 = temp[idx + 512];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 512] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x146(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q13']	['q12']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4096ll) + 4096 + 8192ll * (0) + 16384ll * ((local_index / 4096ll) % 4ll);
		std::complex<double> t0 = temp[idx + 8192];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8192] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x147(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q7']	['q6']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 64ll) + 64 + 128ll * (0) + 256ll * ((local_index / 64ll) % 256ll);
		std::complex<double> t0 = temp[idx + 128];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 128] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x148(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q11']	['q10']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 1024ll) + 1024 + 2048ll * (0) + 4096ll * ((local_index / 1024ll) % 16ll);
		std::complex<double> t0 = temp[idx + 2048];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2048] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x149(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q15']	['q14']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 16384 + 32768ll * (0) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 32768];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32768] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x150(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q6']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 64ll) + 128ll * ((local_index / 64ll) % 512ll);
		std::complex<double> t0 = temp[idx + 0 * 64];
		std::complex<double> t1 = temp[idx + 1 * 64];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 64] = std::complex<double>(t20, t21);
		temp[idx + 1 * 64] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x151(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q7']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 128ll) + 256ll * ((local_index / 128ll) % 256ll);
		std::complex<double> t0 = temp[idx + 0 * 128];
		std::complex<double> t1 = temp[idx + 1 * 128];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 128] = std::complex<double>(t20, t21);
		temp[idx + 1 * 128] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x152(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q10']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 1024ll) + 2048ll * ((local_index / 1024ll) % 32ll);
		std::complex<double> t0 = temp[idx + 0 * 1024];
		std::complex<double> t1 = temp[idx + 1 * 1024];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 1024] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1024] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x153(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q11']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 2048ll) + 4096ll * ((local_index / 2048ll) % 16ll);
		std::complex<double> t0 = temp[idx + 0 * 2048];
		std::complex<double> t1 = temp[idx + 1 * 2048];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 2048] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2048] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x154(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q14']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 16384ll) + 32768ll * ((local_index / 16384ll) % 2ll);
		std::complex<double> t0 = temp[idx + 0 * 16384];
		std::complex<double> t1 = temp[idx + 1 * 16384];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 16384] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16384] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x155(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q15']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 32768ll) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 0 * 32768];
		std::complex<double> t1 = temp[idx + 1 * 32768];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 32768] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32768] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x156(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q5']	['q4']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 16 + 32ll * (0) + 64ll * ((local_index / 16ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x157(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (0) + 2ll * (local_index % 32768ll);
		std::complex<double> t0 = temp[idx + 0 * 1];
		std::complex<double> t1 = temp[idx + 1 * 1];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 1] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x158(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 16384ll);
		std::complex<double> t0 = temp[idx + 0 * 2];
		std::complex<double> t1 = temp[idx + 1 * 2];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 2] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x159(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 8192ll);
		std::complex<double> t0 = temp[idx + 0 * 4];
		std::complex<double> t1 = temp[idx + 1 * 4];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 4] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x160(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * ((local_index / 8ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 0 * 8];
		std::complex<double> t1 = temp[idx + 1 * 8];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 8] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x161(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q8']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 256ll) + 512ll * ((local_index / 256ll) % 128ll);
		std::complex<double> t0 = temp[idx + 0 * 256];
		std::complex<double> t1 = temp[idx + 1 * 256];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 256] = std::complex<double>(t20, t21);
		temp[idx + 1 * 256] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x162(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q9']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 512ll) + 1024ll * ((local_index / 512ll) % 64ll);
		std::complex<double> t0 = temp[idx + 0 * 512];
		std::complex<double> t1 = temp[idx + 1 * 512];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 512] = std::complex<double>(t20, t21);
		temp[idx + 1 * 512] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x163(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q12']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 4096ll) + 8192ll * ((local_index / 4096ll) % 8ll);
		std::complex<double> t0 = temp[idx + 0 * 4096];
		std::complex<double> t1 = temp[idx + 1 * 4096];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 4096] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4096] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x164(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q13']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 8192ll) + 16384ll * ((local_index / 8192ll) % 4ll);
		std::complex<double> t0 = temp[idx + 8192];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8192] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x165(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q7']	['q6']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 64ll) + 64 + 128ll * (0) + 256ll * ((local_index / 64ll) % 256ll);
		std::complex<double> t0 = temp[idx + 128];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 128] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x166(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q11']	['q10']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 1024ll) + 1024 + 2048ll * (0) + 4096ll * ((local_index / 1024ll) % 16ll);
		std::complex<double> t0 = temp[idx + 2048];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2048] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x167(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q15']	['q14']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 16384 + 32768ll * (0) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 32768];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32768] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x168(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q4']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 2048ll);
		std::complex<double> t0 = temp[idx + 0 * 16];
		std::complex<double> t1 = temp[idx + 1 * 16];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 16] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x169(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q5']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 32ll) + 64ll * ((local_index / 32ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 0 * 32];
		std::complex<double> t1 = temp[idx + 1 * 32];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 32] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x170(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q1']	['q0']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (0) + 1 + 2ll * (0) + 4ll * (local_index % 16384ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x171(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q3']	['q2']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 4 + 8ll * (0) + 16ll * ((local_index / 4ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x172(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q9']	['q8']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 256 + 512ll * (0) + 1024ll * ((local_index / 256ll) % 64ll);
		std::complex<double> t0 = temp[idx + 512];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 512] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x173(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q13']	['q12']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4096ll) + 4096 + 8192ll * (0) + 16384ll * ((local_index / 4096ll) % 4ll);
		std::complex<double> t0 = temp[idx + 8192];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8192] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x174(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q10']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 1024ll) + 2048ll * ((local_index / 1024ll) % 32ll);
		std::complex<double> t0 = temp[idx + 0 * 1024];
		std::complex<double> t1 = temp[idx + 1 * 1024];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 1024] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1024] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x175(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q11']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 2048ll) + 4096ll * ((local_index / 2048ll) % 16ll);
		std::complex<double> t0 = temp[idx + 0 * 2048];
		std::complex<double> t1 = temp[idx + 1 * 2048];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 2048] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2048] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x176(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q14']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 16384ll) + 32768ll * ((local_index / 16384ll) % 2ll);
		std::complex<double> t0 = temp[idx + 0 * 16384];
		std::complex<double> t1 = temp[idx + 1 * 16384];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 16384] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16384] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x177(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q15']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 32768ll) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 0 * 32768];
		std::complex<double> t1 = temp[idx + 1 * 32768];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 32768] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32768] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x178(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q5']	['q4']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 16 + 32ll * (0) + 64ll * ((local_index / 16ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x179(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (0) + 2ll * (local_index % 32768ll);
		std::complex<double> t0 = temp[idx + 1];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 1] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x180(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 16384ll);
		std::complex<double> t0 = temp[idx + 0 * 2];
		std::complex<double> t1 = temp[idx + 1 * 2];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 2] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x181(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 8192ll);
		std::complex<double> t0 = temp[idx + 0 * 4];
		std::complex<double> t1 = temp[idx + 1 * 4];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 4] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x182(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * ((local_index / 8ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 0 * 8];
		std::complex<double> t1 = temp[idx + 1 * 8];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 8] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x183(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q8']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 256ll) + 512ll * ((local_index / 256ll) % 128ll);
		std::complex<double> t0 = temp[idx + 0 * 256];
		std::complex<double> t1 = temp[idx + 1 * 256];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 256] = std::complex<double>(t20, t21);
		temp[idx + 1 * 256] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x184(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q9']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 512ll) + 1024ll * ((local_index / 512ll) % 64ll);
		std::complex<double> t0 = temp[idx + 0 * 512];
		std::complex<double> t1 = temp[idx + 1 * 512];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 512] = std::complex<double>(t20, t21);
		temp[idx + 1 * 512] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x185(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q12']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 4096ll) + 8192ll * ((local_index / 4096ll) % 8ll);
		std::complex<double> t0 = temp[idx + 0 * 4096];
		std::complex<double> t1 = temp[idx + 1 * 4096];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 4096] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4096] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x186(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q13']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 8192ll) + 16384ll * ((local_index / 8192ll) % 4ll);
		std::complex<double> t0 = temp[idx + 0 * 8192];
		std::complex<double> t1 = temp[idx + 1 * 8192];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 8192] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8192] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x187(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q6']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 64ll) + 128ll * ((local_index / 64ll) % 512ll);
		std::complex<double> t0 = temp[idx + 0 * 64];
		std::complex<double> t1 = temp[idx + 1 * 64];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 64] = std::complex<double>(t20, t21);
		temp[idx + 1 * 64] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x188(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q7']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 128ll) + 256ll * ((local_index / 128ll) % 256ll);
		std::complex<double> t0 = temp[idx + 0 * 128];
		std::complex<double> t1 = temp[idx + 1 * 128];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 128] = std::complex<double>(t20, t21);
		temp[idx + 1 * 128] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x189(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q7']	['q6']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 64ll) + 64 + 128ll * (0) + 256ll * ((local_index / 64ll) % 256ll);
		std::complex<double> t0 = temp[idx + 128];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 128] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x190(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q11']	['q10']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 1024ll) + 1024 + 2048ll * (0) + 4096ll * ((local_index / 1024ll) % 16ll);
		std::complex<double> t0 = temp[idx + 2048];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2048] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x191(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q15']	['q14']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 16384 + 32768ll * (0) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 32768];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32768] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x192(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q4']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 2048ll);
		std::complex<double> t0 = temp[idx + 0 * 16];
		std::complex<double> t1 = temp[idx + 1 * 16];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 16] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x193(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q5']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 32ll) + 64ll * ((local_index / 32ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 0 * 32];
		std::complex<double> t1 = temp[idx + 1 * 32];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 32] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x194(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q1']	['q0']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (0) + 1 + 2ll * (0) + 4ll * (local_index % 16384ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x195(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q3']	['q2']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 4 + 8ll * (0) + 16ll * ((local_index / 4ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x196(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q9']	['q8']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 256 + 512ll * (0) + 1024ll * ((local_index / 256ll) % 64ll);
		std::complex<double> t0 = temp[idx + 512];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 512] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x197(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q13']	['q12']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4096ll) + 4096 + 8192ll * (0) + 16384ll * ((local_index / 4096ll) % 4ll);
		std::complex<double> t0 = temp[idx + 8192];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8192] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x198(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 8ll) + 16ll * ((local_index / 8ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 0 * 8];
		std::complex<double> t1 = temp[idx + 1 * 8];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 8] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x199(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q8']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 512ll * ((local_index / 256ll) % 128ll);
		std::complex<double> t0 = temp[idx + 256];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 256] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x200(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q9']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 512ll) + 1024ll * ((local_index / 512ll) % 64ll);
		std::complex<double> t0 = temp[idx + 0 * 512];
		std::complex<double> t1 = temp[idx + 1 * 512];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 512] = std::complex<double>(t20, t21);
		temp[idx + 1 * 512] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x201(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q12']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 4096ll) + 8192ll * ((local_index / 4096ll) % 8ll);
		std::complex<double> t0 = temp[idx + 0 * 4096];
		std::complex<double> t1 = temp[idx + 1 * 4096];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 4096] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4096] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x202(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q13']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 8192ll) + 16384ll * ((local_index / 8192ll) % 4ll);
		std::complex<double> t0 = temp[idx + 0 * 8192];
		std::complex<double> t1 = temp[idx + 1 * 8192];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 8192] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8192] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x203(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q6']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 64ll) + 128ll * ((local_index / 64ll) % 512ll);
		std::complex<double> t0 = temp[idx + 0 * 64];
		std::complex<double> t1 = temp[idx + 1 * 64];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 64] = std::complex<double>(t20, t21);
		temp[idx + 1 * 64] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x204(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q7']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 128ll) + 256ll * ((local_index / 128ll) % 256ll);
		std::complex<double> t0 = temp[idx + 0 * 128];
		std::complex<double> t1 = temp[idx + 1 * 128];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 128] = std::complex<double>(t20, t21);
		temp[idx + 1 * 128] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x205(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q10']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 1024ll) + 2048ll * ((local_index / 1024ll) % 32ll);
		std::complex<double> t0 = temp[idx + 0 * 1024];
		std::complex<double> t1 = temp[idx + 1 * 1024];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 1024] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1024] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x206(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q11']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 2048ll) + 4096ll * ((local_index / 2048ll) % 16ll);
		std::complex<double> t0 = temp[idx + 0 * 2048];
		std::complex<double> t1 = temp[idx + 1 * 2048];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 2048] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2048] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x207(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q14']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 16384ll) + 32768ll * ((local_index / 16384ll) % 2ll);
		std::complex<double> t0 = temp[idx + 0 * 16384];
		std::complex<double> t1 = temp[idx + 1 * 16384];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 16384] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16384] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x208(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q15']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 32768ll) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 0 * 32768];
		std::complex<double> t1 = temp[idx + 1 * 32768];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 32768] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32768] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x209(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q5']	['q4']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 16 + 32ll * (0) + 64ll * ((local_index / 16ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x210(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (0) + 2ll * (local_index % 32768ll);
		std::complex<double> t0 = temp[idx + 0 * 1];
		std::complex<double> t1 = temp[idx + 1 * 1];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 1] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x211(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 16384ll);
		std::complex<double> t0 = temp[idx + 0 * 2];
		std::complex<double> t1 = temp[idx + 1 * 2];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 2] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x212(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 8192ll);
		std::complex<double> t0 = temp[idx + 0 * 4];
		std::complex<double> t1 = temp[idx + 1 * 4];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 4] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x213(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q7']	['q6']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 64ll) + 64 + 128ll * (0) + 256ll * ((local_index / 64ll) % 256ll);
		std::complex<double> t0 = temp[idx + 128];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 128] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x214(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q11']	['q10']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 1024ll) + 1024 + 2048ll * (0) + 4096ll * ((local_index / 1024ll) % 16ll);
		std::complex<double> t0 = temp[idx + 2048];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2048] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x215(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q15']	['q14']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 16384 + 32768ll * (0) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 32768];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32768] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x216(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q4']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 2048ll);
		std::complex<double> t0 = temp[idx + 0 * 16];
		std::complex<double> t1 = temp[idx + 1 * 16];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 16] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x217(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q5']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 32ll) + 64ll * ((local_index / 32ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 0 * 32];
		std::complex<double> t1 = temp[idx + 1 * 32];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 32] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x218(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q1']	['q0']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (0) + 1 + 2ll * (0) + 4ll * (local_index % 16384ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x219(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q3']	['q2']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 4 + 8ll * (0) + 16ll * ((local_index / 4ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x220(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q9']	['q8']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 256 + 512ll * (0) + 1024ll * ((local_index / 256ll) % 64ll);
		std::complex<double> t0 = temp[idx + 512];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 512] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x221(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q13']	['q12']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4096ll) + 4096 + 8192ll * (0) + 16384ll * ((local_index / 4096ll) % 4ll);
		std::complex<double> t0 = temp[idx + 8192];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8192] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x222(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q6']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 64ll) + 128ll * ((local_index / 64ll) % 512ll);
		std::complex<double> t0 = temp[idx + 0 * 64];
		std::complex<double> t1 = temp[idx + 1 * 64];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 64] = std::complex<double>(t20, t21);
		temp[idx + 1 * 64] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x223(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q7']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 128ll) + 256ll * ((local_index / 128ll) % 256ll);
		std::complex<double> t0 = temp[idx + 0 * 128];
		std::complex<double> t1 = temp[idx + 1 * 128];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 128] = std::complex<double>(t20, t21);
		temp[idx + 1 * 128] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x224(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q10']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 1024ll) + 2048ll * ((local_index / 1024ll) % 32ll);
		std::complex<double> t0 = temp[idx + 0 * 1024];
		std::complex<double> t1 = temp[idx + 1 * 1024];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 1024] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1024] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x225(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q11']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 2048ll) + 4096ll * ((local_index / 2048ll) % 16ll);
		std::complex<double> t0 = temp[idx + 0 * 2048];
		std::complex<double> t1 = temp[idx + 1 * 2048];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 2048] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2048] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x226(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q14']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 16384ll) + 32768ll * ((local_index / 16384ll) % 2ll);
		std::complex<double> t0 = temp[idx + 0 * 16384];
		std::complex<double> t1 = temp[idx + 1 * 16384];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 16384] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16384] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x227(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q15']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 32768ll) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 0 * 32768];
		std::complex<double> t1 = temp[idx + 1 * 32768];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 32768] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32768] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x228(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q5']	['q4']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 16 + 32ll * (0) + 64ll * ((local_index / 16ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x229(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (0) + 2ll * (local_index % 32768ll);
		std::complex<double> t0 = temp[idx + 0 * 1];
		std::complex<double> t1 = temp[idx + 1 * 1];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 1] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x230(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 16384ll);
		std::complex<double> t0 = temp[idx + 0 * 2];
		std::complex<double> t1 = temp[idx + 1 * 2];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 2] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x231(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 8192ll);
		std::complex<double> t0 = temp[idx + 0 * 4];
		std::complex<double> t1 = temp[idx + 1 * 4];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 4] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x232(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * ((local_index / 8ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 0 * 8];
		std::complex<double> t1 = temp[idx + 1 * 8];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 8] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x233(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q8']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 256ll) + 512ll * ((local_index / 256ll) % 128ll);
		std::complex<double> t0 = temp[idx + 0 * 256];
		std::complex<double> t1 = temp[idx + 1 * 256];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 256] = std::complex<double>(t20, t21);
		temp[idx + 1 * 256] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x234(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q9']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 512ll) + 1024ll * ((local_index / 512ll) % 64ll);
		std::complex<double> t0 = temp[idx + 0 * 512];
		std::complex<double> t1 = temp[idx + 1 * 512];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 512] = std::complex<double>(t20, t21);
		temp[idx + 1 * 512] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x235(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q12']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 4096ll) + 8192ll * ((local_index / 4096ll) % 8ll);
		std::complex<double> t0 = temp[idx + 0 * 4096];
		std::complex<double> t1 = temp[idx + 1 * 4096];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 4096] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4096] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x236(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q13']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 8192ll) + 16384ll * ((local_index / 8192ll) % 4ll);
		std::complex<double> t0 = temp[idx + 0 * 8192];
		std::complex<double> t1 = temp[idx + 1 * 8192];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 8192] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8192] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x237(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q4']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 2048ll);
		std::complex<double> t0 = temp[idx + 16];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 16] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x238(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q5']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 32ll) + 64ll * ((local_index / 32ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x239(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q1']	['q0']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (0) + 1 + 2ll * (0) + 4ll * (local_index % 16384ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x240(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q3']	['q2']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 4 + 8ll * (0) + 16ll * ((local_index / 4ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x241(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q9']	['q8']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 256 + 512ll * (0) + 1024ll * ((local_index / 256ll) % 64ll);
		std::complex<double> t0 = temp[idx + 512];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 512] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x242(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q13']	['q12']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4096ll) + 4096 + 8192ll * (0) + 16384ll * ((local_index / 4096ll) % 4ll);
		std::complex<double> t0 = temp[idx + 8192];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8192] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x243(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q7']	['q6']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 64ll) + 64 + 128ll * (0) + 256ll * ((local_index / 64ll) % 256ll);
		std::complex<double> t0 = temp[idx + 128];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 128] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x244(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q11']	['q10']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 1024ll) + 1024 + 2048ll * (0) + 4096ll * ((local_index / 1024ll) % 16ll);
		std::complex<double> t0 = temp[idx + 2048];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2048] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x245(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q15']	['q14']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 16384 + 32768ll * (0) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 32768];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32768] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x246(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q6']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 64ll) + 128ll * ((local_index / 64ll) % 512ll);
		std::complex<double> t0 = temp[idx + 0 * 64];
		std::complex<double> t1 = temp[idx + 1 * 64];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 64] = std::complex<double>(t20, t21);
		temp[idx + 1 * 64] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x247(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q7']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 128ll) + 256ll * ((local_index / 128ll) % 256ll);
		std::complex<double> t0 = temp[idx + 0 * 128];
		std::complex<double> t1 = temp[idx + 1 * 128];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 128] = std::complex<double>(t20, t21);
		temp[idx + 1 * 128] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x248(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q10']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 1024ll) + 2048ll * ((local_index / 1024ll) % 32ll);
		std::complex<double> t0 = temp[idx + 0 * 1024];
		std::complex<double> t1 = temp[idx + 1 * 1024];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 1024] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1024] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x249(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q11']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 2048ll) + 4096ll * ((local_index / 2048ll) % 16ll);
		std::complex<double> t0 = temp[idx + 0 * 2048];
		std::complex<double> t1 = temp[idx + 1 * 2048];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 2048] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2048] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x250(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q14']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 16384ll) + 32768ll * ((local_index / 16384ll) % 2ll);
		std::complex<double> t0 = temp[idx + 0 * 16384];
		std::complex<double> t1 = temp[idx + 1 * 16384];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 16384] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16384] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x251(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q15']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 32768ll) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 0 * 32768];
		std::complex<double> t1 = temp[idx + 1 * 32768];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 32768] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32768] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x252(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q5']	['q4']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 16 + 32ll * (0) + 64ll * ((local_index / 16ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x253(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (0) + 2ll * (local_index % 32768ll);
		std::complex<double> t0 = temp[idx + 0 * 1];
		std::complex<double> t1 = temp[idx + 1 * 1];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 1] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x254(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 16384ll);
		std::complex<double> t0 = temp[idx + 0 * 2];
		std::complex<double> t1 = temp[idx + 1 * 2];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 2] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x255(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 8192ll);
		std::complex<double> t0 = temp[idx + 0 * 4];
		std::complex<double> t1 = temp[idx + 1 * 4];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 4] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x256(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * ((local_index / 8ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 0 * 8];
		std::complex<double> t1 = temp[idx + 1 * 8];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 8] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x257(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q8']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 256ll) + 512ll * ((local_index / 256ll) % 128ll);
		std::complex<double> t0 = temp[idx + 0 * 256];
		std::complex<double> t1 = temp[idx + 1 * 256];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 256] = std::complex<double>(t20, t21);
		temp[idx + 1 * 256] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x258(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q9']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 512ll) + 1024ll * ((local_index / 512ll) % 64ll);
		std::complex<double> t0 = temp[idx + 0 * 512];
		std::complex<double> t1 = temp[idx + 1 * 512];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 512] = std::complex<double>(t20, t21);
		temp[idx + 1 * 512] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x259(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q12']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 4096ll) + 8192ll * ((local_index / 4096ll) % 8ll);
		std::complex<double> t0 = temp[idx + 0 * 4096];
		std::complex<double> t1 = temp[idx + 1 * 4096];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 4096] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4096] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x260(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q13']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 8192ll) + 16384ll * ((local_index / 8192ll) % 4ll);
		std::complex<double> t0 = temp[idx + 0 * 8192];
		std::complex<double> t1 = temp[idx + 1 * 8192];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 8192] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8192] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x261(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q7']	['q6']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 64ll) + 64 + 128ll * (0) + 256ll * ((local_index / 64ll) % 256ll);
		std::complex<double> t0 = temp[idx + 128];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 128] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x262(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q11']	['q10']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 1024ll) + 1024 + 2048ll * (0) + 4096ll * ((local_index / 1024ll) % 16ll);
		std::complex<double> t0 = temp[idx + 2048];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2048] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x263(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q15']	['q14']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 16384 + 32768ll * (0) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 32768];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32768] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x264(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q4']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 2048ll);
		std::complex<double> t0 = temp[idx + 16];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 16] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x265(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q5']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 32ll) + 64ll * ((local_index / 32ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x266(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q1']	['q0']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (0) + 1 + 2ll * (0) + 4ll * (local_index % 16384ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x267(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q3']	['q2']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 4 + 8ll * (0) + 16ll * ((local_index / 4ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x268(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q9']	['q8']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 256 + 512ll * (0) + 1024ll * ((local_index / 256ll) % 64ll);
		std::complex<double> t0 = temp[idx + 512];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 512] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x269(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q13']	['q12']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4096ll) + 4096 + 8192ll * (0) + 16384ll * ((local_index / 4096ll) % 4ll);
		std::complex<double> t0 = temp[idx + 8192];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8192] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x270(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q6']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 64ll) + 128ll * ((local_index / 64ll) % 512ll);
		std::complex<double> t0 = temp[idx + 0 * 64];
		std::complex<double> t1 = temp[idx + 1 * 64];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 64] = std::complex<double>(t20, t21);
		temp[idx + 1 * 64] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x271(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q7']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 128ll) + 256ll * ((local_index / 128ll) % 256ll);
		std::complex<double> t0 = temp[idx + 0 * 128];
		std::complex<double> t1 = temp[idx + 1 * 128];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 128] = std::complex<double>(t20, t21);
		temp[idx + 1 * 128] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x272(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q10']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 1024ll) + 2048ll * ((local_index / 1024ll) % 32ll);
		std::complex<double> t0 = temp[idx + 0 * 1024];
		std::complex<double> t1 = temp[idx + 1 * 1024];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 1024] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1024] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x273(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q11']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2048ll) + 4096ll * ((local_index / 2048ll) % 16ll);
		std::complex<double> t0 = temp[idx + 2048];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2048] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x274(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q14']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 16384ll) + 32768ll * ((local_index / 16384ll) % 2ll);
		std::complex<double> t0 = temp[idx + 0 * 16384];
		std::complex<double> t1 = temp[idx + 1 * 16384];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 16384] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16384] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x275(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q15']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 32768ll) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 0 * 32768];
		std::complex<double> t1 = temp[idx + 1 * 32768];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 32768] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32768] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x276(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q5']	['q4']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 16 + 32ll * (0) + 64ll * ((local_index / 16ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x277(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (0) + 2ll * (local_index % 32768ll);
		std::complex<double> t0 = temp[idx + 0 * 1];
		std::complex<double> t1 = temp[idx + 1 * 1];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 1] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x278(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 16384ll);
		std::complex<double> t0 = temp[idx + 0 * 2];
		std::complex<double> t1 = temp[idx + 1 * 2];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 2] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x279(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 8192ll);
		std::complex<double> t0 = temp[idx + 0 * 4];
		std::complex<double> t1 = temp[idx + 1 * 4];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 4] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x280(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * ((local_index / 8ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 0 * 8];
		std::complex<double> t1 = temp[idx + 1 * 8];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 8] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x281(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q8']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 512ll * ((local_index / 256ll) % 128ll);
		std::complex<double> t0 = temp[idx + 256];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 256] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x282(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q9']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 512ll) + 1024ll * ((local_index / 512ll) % 64ll);
		std::complex<double> t0 = temp[idx + 0 * 512];
		std::complex<double> t1 = temp[idx + 1 * 512];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 512] = std::complex<double>(t20, t21);
		temp[idx + 1 * 512] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x283(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q12']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 4096ll) + 8192ll * ((local_index / 4096ll) % 8ll);
		std::complex<double> t0 = temp[idx + 0 * 4096];
		std::complex<double> t1 = temp[idx + 1 * 4096];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 4096] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4096] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x284(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q13']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 8192ll) + 16384ll * ((local_index / 8192ll) % 4ll);
		std::complex<double> t0 = temp[idx + 0 * 8192];
		std::complex<double> t1 = temp[idx + 1 * 8192];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 8192] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8192] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x285(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q12']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 4096ll) + 8192ll * ((local_index / 4096ll) % 8ll);
		std::complex<double> t0 = temp[idx + 0 * 4096];
		std::complex<double> t1 = temp[idx + 1 * 4096];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 4096] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4096] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x286(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q13']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 8192ll) + 16384ll * ((local_index / 8192ll) % 4ll);
		std::complex<double> t0 = temp[idx + 0 * 8192];
		std::complex<double> t1 = temp[idx + 1 * 8192];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 8192] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8192] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x287(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q7']	['q6']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 64ll) + 64 + 128ll * (0) + 256ll * ((local_index / 64ll) % 256ll);
		std::complex<double> t0 = temp[idx + 128];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 128] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x288(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q11']	['q10']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 1024ll) + 1024 + 2048ll * (0) + 4096ll * ((local_index / 1024ll) % 16ll);
		std::complex<double> t0 = temp[idx + 2048];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2048] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x289(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q15']	['q14']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 16384 + 32768ll * (0) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 32768];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32768] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x290(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q4']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 2048ll);
		std::complex<double> t0 = temp[idx + 0 * 16];
		std::complex<double> t1 = temp[idx + 1 * 16];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 16] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x291(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q5']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 32ll) + 64ll * ((local_index / 32ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 0 * 32];
		std::complex<double> t1 = temp[idx + 1 * 32];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 32] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x292(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q1']	['q0']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (0) + 1 + 2ll * (0) + 4ll * (local_index % 16384ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x293(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q3']	['q2']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 4 + 8ll * (0) + 16ll * ((local_index / 4ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x294(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q9']	['q8']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 256 + 512ll * (0) + 1024ll * ((local_index / 256ll) % 64ll);
		std::complex<double> t0 = temp[idx + 512];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 512] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x295(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 16384ll);
		std::complex<double> t0 = temp[idx + 0 * 2];
		std::complex<double> t1 = temp[idx + 1 * 2];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 2] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x296(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 8192ll);
		std::complex<double> t0 = temp[idx + 0 * 4];
		std::complex<double> t1 = temp[idx + 1 * 4];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 4] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x297(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 8ll) + 16ll * ((local_index / 8ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x298(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q8']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 512ll * ((local_index / 256ll) % 128ll);
		std::complex<double> t0 = temp[idx + 256];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 256] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x299(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q9']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 512ll) + 1024ll * ((local_index / 512ll) % 64ll);
		std::complex<double> t0 = temp[idx + 0 * 512];
		std::complex<double> t1 = temp[idx + 1 * 512];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 512] = std::complex<double>(t20, t21);
		temp[idx + 1 * 512] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x300(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q13']	['q12']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4096ll) + 4096 + 8192ll * (0) + 16384ll * ((local_index / 4096ll) % 4ll);
		std::complex<double> t0 = temp[idx + 8192];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8192] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x301(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q6']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 64ll) + 128ll * ((local_index / 64ll) % 512ll);
		std::complex<double> t0 = temp[idx + 0 * 64];
		std::complex<double> t1 = temp[idx + 1 * 64];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 64] = std::complex<double>(t20, t21);
		temp[idx + 1 * 64] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x302(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q7']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 128ll) + 256ll * ((local_index / 128ll) % 256ll);
		std::complex<double> t0 = temp[idx + 128];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 128] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x303(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q10']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 1024ll) + 2048ll * ((local_index / 1024ll) % 32ll);
		std::complex<double> t0 = temp[idx + 1024];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 1024] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x304(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q11']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2048ll) + 4096ll * ((local_index / 2048ll) % 16ll);
		std::complex<double> t0 = temp[idx + 2048];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2048] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x305(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q14']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 16384ll) + 32768ll * ((local_index / 16384ll) % 2ll);
		std::complex<double> t0 = temp[idx + 0 * 16384];
		std::complex<double> t1 = temp[idx + 1 * 16384];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 16384] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16384] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x306(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q15']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 32768ll) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 0 * 32768];
		std::complex<double> t1 = temp[idx + 1 * 32768];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 32768] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32768] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x307(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q5']	['q4']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 16 + 32ll * (0) + 64ll * ((local_index / 16ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x308(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (0) + 2ll * (local_index % 32768ll);
		std::complex<double> t0 = temp[idx + 0 * 1];
		std::complex<double> t1 = temp[idx + 1 * 1];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 1] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x309(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q12']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 4096ll) + 8192ll * ((local_index / 4096ll) % 8ll);
		std::complex<double> t0 = temp[idx + 0 * 4096];
		std::complex<double> t1 = temp[idx + 1 * 4096];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 4096] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4096] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x310(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q13']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 8192ll) + 16384ll * ((local_index / 8192ll) % 4ll);
		std::complex<double> t0 = temp[idx + 0 * 8192];
		std::complex<double> t1 = temp[idx + 1 * 8192];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 8192] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8192] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x311(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q7']	['q6']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 64ll) + 64 + 128ll * (0) + 256ll * ((local_index / 64ll) % 256ll);
		std::complex<double> t0 = temp[idx + 128];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 128] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x312(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q11']	['q10']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 1024ll) + 1024 + 2048ll * (0) + 4096ll * ((local_index / 1024ll) % 16ll);
		std::complex<double> t0 = temp[idx + 2048];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2048] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x313(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q15']	['q14']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 16384 + 32768ll * (0) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 32768];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32768] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x314(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q4']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 2048ll);
		std::complex<double> t0 = temp[idx + 0 * 16];
		std::complex<double> t1 = temp[idx + 1 * 16];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 16] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x315(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q5']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 32ll) + 64ll * ((local_index / 32ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x316(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q1']	['q0']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (0) + 1 + 2ll * (0) + 4ll * (local_index % 16384ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x317(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q3']	['q2']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 4 + 8ll * (0) + 16ll * ((local_index / 4ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x318(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q9']	['q8']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 256 + 512ll * (0) + 1024ll * ((local_index / 256ll) % 64ll);
		std::complex<double> t0 = temp[idx + 512];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 512] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x319(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (0) + 2ll * (local_index % 32768ll);
		std::complex<double> t0 = temp[idx + 0 * 1];
		std::complex<double> t1 = temp[idx + 1 * 1];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 1] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x320(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 16384ll);
		std::complex<double> t0 = temp[idx + 0 * 2];
		std::complex<double> t1 = temp[idx + 1 * 2];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 2] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x321(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q13']	['q12']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4096ll) + 4096 + 8192ll * (0) + 16384ll * ((local_index / 4096ll) % 4ll);
		std::complex<double> t0 = temp[idx + 8192];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8192] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x322(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q6']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 64ll) + 128ll * ((local_index / 64ll) % 512ll);
		std::complex<double> t0 = temp[idx + 0 * 64];
		std::complex<double> t1 = temp[idx + 1 * 64];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 64] = std::complex<double>(t20, t21);
		temp[idx + 1 * 64] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x323(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q7']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 128ll) + 256ll * ((local_index / 128ll) % 256ll);
		std::complex<double> t0 = temp[idx + 0 * 128];
		std::complex<double> t1 = temp[idx + 1 * 128];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 128] = std::complex<double>(t20, t21);
		temp[idx + 1 * 128] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x324(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q10']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 1024ll) + 2048ll * ((local_index / 1024ll) % 32ll);
		std::complex<double> t0 = temp[idx + 0 * 1024];
		std::complex<double> t1 = temp[idx + 1 * 1024];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 1024] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1024] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x325(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q11']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 2048ll) + 4096ll * ((local_index / 2048ll) % 16ll);
		std::complex<double> t0 = temp[idx + 0 * 2048];
		std::complex<double> t1 = temp[idx + 1 * 2048];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 2048] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2048] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x326(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q14']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 16384ll) + 32768ll * ((local_index / 16384ll) % 2ll);
		std::complex<double> t0 = temp[idx + 0 * 16384];
		std::complex<double> t1 = temp[idx + 1 * 16384];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 16384] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16384] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x327(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q15']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 32768ll) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 0 * 32768];
		std::complex<double> t1 = temp[idx + 1 * 32768];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 32768] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32768] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x328(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q5']	['q4']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 16 + 32ll * (0) + 64ll * ((local_index / 16ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x329(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 8192ll);
		std::complex<double> t0 = temp[idx + 0 * 4];
		std::complex<double> t1 = temp[idx + 1 * 4];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 4] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x330(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * ((local_index / 8ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 0 * 8];
		std::complex<double> t1 = temp[idx + 1 * 8];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 8] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x331(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q8']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 256ll) + 512ll * ((local_index / 256ll) % 128ll);
		std::complex<double> t0 = temp[idx + 0 * 256];
		std::complex<double> t1 = temp[idx + 1 * 256];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 256] = std::complex<double>(t20, t21);
		temp[idx + 1 * 256] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x332(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q9']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 512ll) + 1024ll * ((local_index / 512ll) % 64ll);
		std::complex<double> t0 = temp[idx + 0 * 512];
		std::complex<double> t1 = temp[idx + 1 * 512];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 512] = std::complex<double>(t20, t21);
		temp[idx + 1 * 512] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x333(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q13']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 8192ll) + 16384ll * ((local_index / 8192ll) % 4ll);
		std::complex<double> t0 = temp[idx + 0 * 8192];
		std::complex<double> t1 = temp[idx + 1 * 8192];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 8192] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8192] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x334(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q7']	['q6']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 64ll) + 64 + 128ll * (0) + 256ll * ((local_index / 64ll) % 256ll);
		std::complex<double> t0 = temp[idx + 128];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 128] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x335(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q11']	['q10']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 1024ll) + 1024 + 2048ll * (0) + 4096ll * ((local_index / 1024ll) % 16ll);
		std::complex<double> t0 = temp[idx + 2048];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2048] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x336(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q15']	['q14']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 16384 + 32768ll * (0) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 32768];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32768] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x337(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q4']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 2048ll);
		std::complex<double> t0 = temp[idx + 16];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 16] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x338(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q5']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 32ll) + 64ll * ((local_index / 32ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 0 * 32];
		std::complex<double> t1 = temp[idx + 1 * 32];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 32] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x339(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q3']	['q2']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 4 + 8ll * (0) + 16ll * ((local_index / 4ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x340(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q9']	['q8']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 256 + 512ll * (0) + 1024ll * ((local_index / 256ll) % 64ll);
		std::complex<double> t0 = temp[idx + 512];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 512] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x341(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (0) + 2ll * (local_index % 32768ll);
		std::complex<double> t0 = temp[idx + 0 * 1];
		std::complex<double> t1 = temp[idx + 1 * 1];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 1] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x342(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 16384ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x343(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q12']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 4096ll) + 8192ll * ((local_index / 4096ll) % 8ll);
		std::complex<double> t0 = temp[idx + 0 * 4096];
		std::complex<double> t1 = temp[idx + 1 * 4096];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 4096] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4096] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x344(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q1']	['q0']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (0) + 1 + 2ll * (0) + 4ll * (local_index % 16384ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x345(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q13']	['q12']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4096ll) + 4096 + 8192ll * (0) + 16384ll * ((local_index / 4096ll) % 4ll);
		std::complex<double> t0 = temp[idx + 8192];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8192] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x346(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q6']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 64ll) + 128ll * ((local_index / 64ll) % 512ll);
		std::complex<double> t0 = temp[idx + 0 * 64];
		std::complex<double> t1 = temp[idx + 1 * 64];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 64] = std::complex<double>(t20, t21);
		temp[idx + 1 * 64] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x347(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q7']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 128ll) + 256ll * ((local_index / 128ll) % 256ll);
		std::complex<double> t0 = temp[idx + 0 * 128];
		std::complex<double> t1 = temp[idx + 1 * 128];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 128] = std::complex<double>(t20, t21);
		temp[idx + 1 * 128] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x348(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q10']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 1024ll) + 2048ll * ((local_index / 1024ll) % 32ll);
		std::complex<double> t0 = temp[idx + 0 * 1024];
		std::complex<double> t1 = temp[idx + 1 * 1024];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 1024] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1024] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x349(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q11']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 2048ll) + 4096ll * ((local_index / 2048ll) % 16ll);
		std::complex<double> t0 = temp[idx + 0 * 2048];
		std::complex<double> t1 = temp[idx + 1 * 2048];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 2048] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2048] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x350(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q14']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 16384ll) + 32768ll * ((local_index / 16384ll) % 2ll);
		std::complex<double> t0 = temp[idx + 0 * 16384];
		std::complex<double> t1 = temp[idx + 1 * 16384];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 16384] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16384] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x351(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q15']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 32768ll) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 0 * 32768];
		std::complex<double> t1 = temp[idx + 1 * 32768];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 32768] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32768] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x352(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q5']	['q4']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 16 + 32ll * (0) + 64ll * ((local_index / 16ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x353(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 8192ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x354(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 8ll) + 16ll * ((local_index / 8ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 0 * 8];
		std::complex<double> t1 = temp[idx + 1 * 8];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 8] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x355(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q8']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 256ll) + 512ll * ((local_index / 256ll) % 128ll);
		std::complex<double> t0 = temp[idx + 0 * 256];
		std::complex<double> t1 = temp[idx + 1 * 256];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 256] = std::complex<double>(t20, t21);
		temp[idx + 1 * 256] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x356(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q9']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 512ll) + 1024ll * ((local_index / 512ll) % 64ll);
		std::complex<double> t0 = temp[idx + 0 * 512];
		std::complex<double> t1 = temp[idx + 1 * 512];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 512] = std::complex<double>(t20, t21);
		temp[idx + 1 * 512] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x357(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q9']	['q8']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 256 + 512ll * (0) + 1024ll * ((local_index / 256ll) % 64ll);
		std::complex<double> t0 = temp[idx + 512];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 512] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x358(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (0) + 2ll * (local_index % 32768ll);
		std::complex<double> t0 = temp[idx + 1];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 1] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x359(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 16384ll);
		std::complex<double> t0 = temp[idx + 0 * 2];
		std::complex<double> t1 = temp[idx + 1 * 2];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 2] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x360(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q12']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 4096ll) + 8192ll * ((local_index / 4096ll) % 8ll);
		std::complex<double> t0 = temp[idx + 0 * 4096];
		std::complex<double> t1 = temp[idx + 1 * 4096];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 4096] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4096] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x361(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q13']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 8192ll) + 16384ll * ((local_index / 8192ll) % 4ll);
		std::complex<double> t0 = temp[idx + 0 * 8192];
		std::complex<double> t1 = temp[idx + 1 * 8192];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 8192] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8192] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x362(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q7']	['q6']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 64ll) + 64 + 128ll * (0) + 256ll * ((local_index / 64ll) % 256ll);
		std::complex<double> t0 = temp[idx + 128];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 128] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x363(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q11']	['q10']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 1024ll) + 1024 + 2048ll * (0) + 4096ll * ((local_index / 1024ll) % 16ll);
		std::complex<double> t0 = temp[idx + 2048];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2048] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x364(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q15']	['q14']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 16384 + 32768ll * (0) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 32768];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32768] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x365(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q4']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 2048ll);
		std::complex<double> t0 = temp[idx + 0 * 16];
		std::complex<double> t1 = temp[idx + 1 * 16];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 16] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x366(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q5']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 32ll) + 64ll * ((local_index / 32ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x367(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q3']	['q2']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 4 + 8ll * (0) + 16ll * ((local_index / 4ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x368(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q13']	['q12']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4096ll) + 4096 + 8192ll * (0) + 16384ll * ((local_index / 4096ll) % 4ll);
		std::complex<double> t0 = temp[idx + 8192];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8192] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x369(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q6']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 64ll) + 128ll * ((local_index / 64ll) % 512ll);
		std::complex<double> t0 = temp[idx + 64];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 64] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x370(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q7']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 128ll) + 256ll * ((local_index / 128ll) % 256ll);
		std::complex<double> t0 = temp[idx + 0 * 128];
		std::complex<double> t1 = temp[idx + 1 * 128];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 128] = std::complex<double>(t20, t21);
		temp[idx + 1 * 128] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x371(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q10']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 1024ll) + 2048ll * ((local_index / 1024ll) % 32ll);
		std::complex<double> t0 = temp[idx + 0 * 1024];
		std::complex<double> t1 = temp[idx + 1 * 1024];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 1024] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1024] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x372(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q11']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 2048ll) + 4096ll * ((local_index / 2048ll) % 16ll);
		std::complex<double> t0 = temp[idx + 0 * 2048];
		std::complex<double> t1 = temp[idx + 1 * 2048];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 2048] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2048] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x373(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q14']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 16384ll) + 32768ll * ((local_index / 16384ll) % 2ll);
		std::complex<double> t0 = temp[idx + 0 * 16384];
		std::complex<double> t1 = temp[idx + 1 * 16384];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 16384] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16384] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x374(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q15']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 32768ll) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 0 * 32768];
		std::complex<double> t1 = temp[idx + 1 * 32768];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 32768] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32768] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x375(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q5']	['q4']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 16 + 32ll * (0) + 64ll * ((local_index / 16ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x376(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 8192ll);
		std::complex<double> t0 = temp[idx + 0 * 4];
		std::complex<double> t1 = temp[idx + 1 * 4];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 4] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x377(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 8ll) + 16ll * ((local_index / 8ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 0 * 8];
		std::complex<double> t1 = temp[idx + 1 * 8];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 8] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x378(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q8']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 256ll) + 512ll * ((local_index / 256ll) % 128ll);
		std::complex<double> t0 = temp[idx + 0 * 256];
		std::complex<double> t1 = temp[idx + 1 * 256];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 256] = std::complex<double>(t20, t21);
		temp[idx + 1 * 256] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x379(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q9']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 512ll) + 1024ll * ((local_index / 512ll) % 64ll);
		std::complex<double> t0 = temp[idx + 0 * 512];
		std::complex<double> t1 = temp[idx + 1 * 512];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 512] = std::complex<double>(t20, t21);
		temp[idx + 1 * 512] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x380(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q1']	['q0']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (0) + 1 + 2ll * (0) + 4ll * (local_index % 16384ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x381(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (0) + 2ll * (local_index % 32768ll);
		std::complex<double> t0 = temp[idx + 0 * 1];
		std::complex<double> t1 = temp[idx + 1 * 1];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 1] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x382(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 16384ll);
		std::complex<double> t0 = temp[idx + 0 * 2];
		std::complex<double> t1 = temp[idx + 1 * 2];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 2] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x383(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q12']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 4096ll) + 8192ll * ((local_index / 4096ll) % 8ll);
		std::complex<double> t0 = temp[idx + 0 * 4096];
		std::complex<double> t1 = temp[idx + 1 * 4096];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 4096] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4096] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x384(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q13']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 8192ll) + 16384ll * ((local_index / 8192ll) % 4ll);
		std::complex<double> t0 = temp[idx + 0 * 8192];
		std::complex<double> t1 = temp[idx + 1 * 8192];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 8192] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8192] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x385(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q7']	['q6']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 64ll) + 64 + 128ll * (0) + 256ll * ((local_index / 64ll) % 256ll);
		std::complex<double> t0 = temp[idx + 128];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 128] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x386(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q11']	['q10']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 1024ll) + 1024 + 2048ll * (0) + 4096ll * ((local_index / 1024ll) % 16ll);
		std::complex<double> t0 = temp[idx + 2048];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2048] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x387(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q15']	['q14']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 16384 + 32768ll * (0) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 32768];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32768] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x388(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q4']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 2048ll);
		std::complex<double> t0 = temp[idx + 0 * 16];
		std::complex<double> t1 = temp[idx + 1 * 16];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 16] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x389(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q5']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 32ll) + 64ll * ((local_index / 32ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 0 * 32];
		std::complex<double> t1 = temp[idx + 1 * 32];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 32] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x390(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q3']	['q2']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 4 + 8ll * (0) + 16ll * ((local_index / 4ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x391(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q9']	['q8']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 256 + 512ll * (0) + 1024ll * ((local_index / 256ll) % 64ll);
		std::complex<double> t0 = temp[idx + 512];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 512] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x392(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q5']	['q4']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 16 + 32ll * (0) + 64ll * ((local_index / 16ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x393(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 8192ll);
		std::complex<double> t0 = temp[idx + 0 * 4];
		std::complex<double> t1 = temp[idx + 1 * 4];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 4] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x394(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * ((local_index / 8ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 0 * 8];
		std::complex<double> t1 = temp[idx + 1 * 8];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 8] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x395(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q8']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 256ll) + 512ll * ((local_index / 256ll) % 128ll);
		std::complex<double> t0 = temp[idx + 0 * 256];
		std::complex<double> t1 = temp[idx + 1 * 256];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 256] = std::complex<double>(t20, t21);
		temp[idx + 1 * 256] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x396(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q9']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 512ll) + 1024ll * ((local_index / 512ll) % 64ll);
		std::complex<double> t0 = temp[idx + 0 * 512];
		std::complex<double> t1 = temp[idx + 1 * 512];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 512] = std::complex<double>(t20, t21);
		temp[idx + 1 * 512] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x397(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q1']	['q0']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (0) + 1 + 2ll * (0) + 4ll * (local_index % 16384ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x398(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q13']	['q12']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4096ll) + 4096 + 8192ll * (0) + 16384ll * ((local_index / 4096ll) % 4ll);
		std::complex<double> t0 = temp[idx + 8192];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8192] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x399(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q6']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 64ll) + 128ll * ((local_index / 64ll) % 512ll);
		std::complex<double> t0 = temp[idx + 0 * 64];
		std::complex<double> t1 = temp[idx + 1 * 64];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 64] = std::complex<double>(t20, t21);
		temp[idx + 1 * 64] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x400(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q7']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 128ll) + 256ll * ((local_index / 128ll) % 256ll);
		std::complex<double> t0 = temp[idx + 0 * 128];
		std::complex<double> t1 = temp[idx + 1 * 128];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 128] = std::complex<double>(t20, t21);
		temp[idx + 1 * 128] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x401(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q10']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 1024ll) + 2048ll * ((local_index / 1024ll) % 32ll);
		std::complex<double> t0 = temp[idx + 0 * 1024];
		std::complex<double> t1 = temp[idx + 1 * 1024];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 1024] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1024] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x402(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q11']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 2048ll) + 4096ll * ((local_index / 2048ll) % 16ll);
		std::complex<double> t0 = temp[idx + 0 * 2048];
		std::complex<double> t1 = temp[idx + 1 * 2048];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 2048] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2048] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x403(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q14']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 16384ll) + 32768ll * ((local_index / 16384ll) % 2ll);
		std::complex<double> t0 = temp[idx + 0 * 16384];
		std::complex<double> t1 = temp[idx + 1 * 16384];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 16384] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16384] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x404(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q15']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 32768ll) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 0 * 32768];
		std::complex<double> t1 = temp[idx + 1 * 32768];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 32768] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32768] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x405(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (0) + 2ll * (local_index % 32768ll);
		std::complex<double> t0 = temp[idx + 0 * 1];
		std::complex<double> t1 = temp[idx + 1 * 1];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 1] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x406(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 16384ll);
		std::complex<double> t0 = temp[idx + 0 * 2];
		std::complex<double> t1 = temp[idx + 1 * 2];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 2] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x407(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q12']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 4096ll) + 8192ll * ((local_index / 4096ll) % 8ll);
		std::complex<double> t0 = temp[idx + 0 * 4096];
		std::complex<double> t1 = temp[idx + 1 * 4096];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 4096] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4096] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x408(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q13']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 8192ll) + 16384ll * ((local_index / 8192ll) % 4ll);
		std::complex<double> t0 = temp[idx + 0 * 8192];
		std::complex<double> t1 = temp[idx + 1 * 8192];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 8192] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8192] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x409(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q7']	['q6']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 64ll) + 64 + 128ll * (0) + 256ll * ((local_index / 64ll) % 256ll);
		std::complex<double> t0 = temp[idx + 128];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 128] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x410(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q11']	['q10']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 1024ll) + 1024 + 2048ll * (0) + 4096ll * ((local_index / 1024ll) % 16ll);
		std::complex<double> t0 = temp[idx + 2048];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2048] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x411(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q15']	['q14']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 16384 + 32768ll * (0) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 32768];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32768] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x412(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q4']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 2048ll);
		std::complex<double> t0 = temp[idx + 0 * 16];
		std::complex<double> t1 = temp[idx + 1 * 16];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 16] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x413(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q5']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 32ll) + 64ll * ((local_index / 32ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x414(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q3']	['q2']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 4 + 8ll * (0) + 16ll * ((local_index / 4ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x415(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q9']	['q8']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 256 + 512ll * (0) + 1024ll * ((local_index / 256ll) % 64ll);
		std::complex<double> t0 = temp[idx + 512];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 512] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x416(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q1']	['q0']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (0) + 1 + 2ll * (0) + 4ll * (local_index % 16384ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x417(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q13']	['q12']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4096ll) + 4096 + 8192ll * (0) + 16384ll * ((local_index / 4096ll) % 4ll);
		std::complex<double> t0 = temp[idx + 8192];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8192] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x418(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q6']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 64ll) + 128ll * ((local_index / 64ll) % 512ll);
		std::complex<double> t0 = temp[idx + 64];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 64] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x419(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q7']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 128ll) + 256ll * ((local_index / 128ll) % 256ll);
		std::complex<double> t0 = temp[idx + 0 * 128];
		std::complex<double> t1 = temp[idx + 1 * 128];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 128] = std::complex<double>(t20, t21);
		temp[idx + 1 * 128] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x420(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q10']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 1024ll) + 2048ll * ((local_index / 1024ll) % 32ll);
		std::complex<double> t0 = temp[idx + 0 * 1024];
		std::complex<double> t1 = temp[idx + 1 * 1024];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 1024] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1024] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x421(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q11']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 2048ll) + 4096ll * ((local_index / 2048ll) % 16ll);
		std::complex<double> t0 = temp[idx + 0 * 2048];
		std::complex<double> t1 = temp[idx + 1 * 2048];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 2048] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2048] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x422(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q14']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 32768ll * ((local_index / 16384ll) % 2ll);
		std::complex<double> t0 = temp[idx + 16384];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 16384] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x423(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q15']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 32768ll) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 0 * 32768];
		std::complex<double> t1 = temp[idx + 1 * 32768];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 32768] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32768] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x424(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q5']	['q4']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 16 + 32ll * (0) + 64ll * ((local_index / 16ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x425(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 8192ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x426(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * ((local_index / 8ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 0 * 8];
		std::complex<double> t1 = temp[idx + 1 * 8];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 8] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x427(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q8']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 256ll) + 512ll * ((local_index / 256ll) % 128ll);
		std::complex<double> t0 = temp[idx + 0 * 256];
		std::complex<double> t1 = temp[idx + 1 * 256];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 256] = std::complex<double>(t20, t21);
		temp[idx + 1 * 256] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x428(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q9']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 512ll) + 1024ll * ((local_index / 512ll) % 64ll);
		std::complex<double> t0 = temp[idx + 0 * 512];
		std::complex<double> t1 = temp[idx + 1 * 512];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 512] = std::complex<double>(t20, t21);
		temp[idx + 1 * 512] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x429(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q13']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 8192ll) + 16384ll * ((local_index / 8192ll) % 4ll);
		std::complex<double> t0 = temp[idx + 0 * 8192];
		std::complex<double> t1 = temp[idx + 1 * 8192];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 8192] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8192] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x430(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q7']	['q6']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 64ll) + 64 + 128ll * (0) + 256ll * ((local_index / 64ll) % 256ll);
		std::complex<double> t0 = temp[idx + 128];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 128] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x431(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q11']	['q10']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 1024ll) + 1024 + 2048ll * (0) + 4096ll * ((local_index / 1024ll) % 16ll);
		std::complex<double> t0 = temp[idx + 2048];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2048] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x432(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q15']	['q14']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 16384 + 32768ll * (0) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 32768];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32768] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x433(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q4']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 2048ll);
		std::complex<double> t0 = temp[idx + 0 * 16];
		std::complex<double> t1 = temp[idx + 1 * 16];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 16] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x434(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q5']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 32ll) + 64ll * ((local_index / 32ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 0 * 32];
		std::complex<double> t1 = temp[idx + 1 * 32];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 32] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x435(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q3']	['q2']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 4 + 8ll * (0) + 16ll * ((local_index / 4ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x436(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q9']	['q8']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 256 + 512ll * (0) + 1024ll * ((local_index / 256ll) % 64ll);
		std::complex<double> t0 = temp[idx + 512];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 512] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x437(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (0) + 2ll * (local_index % 32768ll);
		std::complex<double> t0 = temp[idx + 0 * 1];
		std::complex<double> t1 = temp[idx + 1 * 1];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 1] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x438(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 16384ll);
		std::complex<double> t0 = temp[idx + 0 * 2];
		std::complex<double> t1 = temp[idx + 1 * 2];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 2] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x439(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q12']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 4096ll) + 8192ll * ((local_index / 4096ll) % 8ll);
		std::complex<double> t0 = temp[idx + 0 * 4096];
		std::complex<double> t1 = temp[idx + 1 * 4096];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 4096] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4096] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x440(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q1']	['q0']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (0) + 1 + 2ll * (0) + 4ll * (local_index % 16384ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x441(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q13']	['q12']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4096ll) + 4096 + 8192ll * (0) + 16384ll * ((local_index / 4096ll) % 4ll);
		std::complex<double> t0 = temp[idx + 8192];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8192] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x442(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q6']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 64ll) + 128ll * ((local_index / 64ll) % 512ll);
		std::complex<double> t0 = temp[idx + 0 * 64];
		std::complex<double> t1 = temp[idx + 1 * 64];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 64] = std::complex<double>(t20, t21);
		temp[idx + 1 * 64] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x443(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q7']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 128ll) + 256ll * ((local_index / 128ll) % 256ll);
		std::complex<double> t0 = temp[idx + 128];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 128] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x444(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q10']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 1024ll) + 2048ll * ((local_index / 1024ll) % 32ll);
		std::complex<double> t0 = temp[idx + 1024];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 1024] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x445(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q11']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 2048ll) + 4096ll * ((local_index / 2048ll) % 16ll);
		std::complex<double> t0 = temp[idx + 0 * 2048];
		std::complex<double> t1 = temp[idx + 1 * 2048];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 2048] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2048] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x446(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q14']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 32768ll * ((local_index / 16384ll) % 2ll);
		std::complex<double> t0 = temp[idx + 16384];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 16384] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x447(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q15']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 32768ll) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 0 * 32768];
		std::complex<double> t1 = temp[idx + 1 * 32768];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 32768] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32768] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x448(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q5']	['q4']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 16 + 32ll * (0) + 64ll * ((local_index / 16ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x449(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 8192ll);
		std::complex<double> t0 = temp[idx + 0 * 4];
		std::complex<double> t1 = temp[idx + 1 * 4];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 4] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x450(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * ((local_index / 8ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 0 * 8];
		std::complex<double> t1 = temp[idx + 1 * 8];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 8] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x451(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q8']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 256ll) + 512ll * ((local_index / 256ll) % 128ll);
		std::complex<double> t0 = temp[idx + 0 * 256];
		std::complex<double> t1 = temp[idx + 1 * 256];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 256] = std::complex<double>(t20, t21);
		temp[idx + 1 * 256] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x452(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q9']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 512ll) + 1024ll * ((local_index / 512ll) % 64ll);
		std::complex<double> t0 = temp[idx + 0 * 512];
		std::complex<double> t1 = temp[idx + 1 * 512];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 512] = std::complex<double>(t20, t21);
		temp[idx + 1 * 512] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x453(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q9']	['q8']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 256 + 512ll * (0) + 1024ll * ((local_index / 256ll) % 64ll);
		std::complex<double> t0 = temp[idx + 512];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 512] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x454(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (0) + 2ll * (local_index % 32768ll);
		std::complex<double> t0 = temp[idx + 1];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 1] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x455(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 16384ll);
		std::complex<double> t0 = temp[idx + 0 * 2];
		std::complex<double> t1 = temp[idx + 1 * 2];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 2] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x456(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q12']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 4096ll) + 8192ll * ((local_index / 4096ll) % 8ll);
		std::complex<double> t0 = temp[idx + 0 * 4096];
		std::complex<double> t1 = temp[idx + 1 * 4096];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 4096] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4096] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x457(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q13']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 8192ll) + 16384ll * ((local_index / 8192ll) % 4ll);
		std::complex<double> t0 = temp[idx + 8192];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8192] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x458(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q7']	['q6']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 64ll) + 64 + 128ll * (0) + 256ll * ((local_index / 64ll) % 256ll);
		std::complex<double> t0 = temp[idx + 128];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 128] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x459(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q11']	['q10']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 1024ll) + 1024 + 2048ll * (0) + 4096ll * ((local_index / 1024ll) % 16ll);
		std::complex<double> t0 = temp[idx + 2048];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2048] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x460(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q15']	['q14']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 16384 + 32768ll * (0) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 32768];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32768] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x461(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q4']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 2048ll);
		std::complex<double> t0 = temp[idx + 0 * 16];
		std::complex<double> t1 = temp[idx + 1 * 16];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 16] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x462(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q5']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 32ll) + 64ll * ((local_index / 32ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 0 * 32];
		std::complex<double> t1 = temp[idx + 1 * 32];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 32] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x463(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q3']	['q2']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 4 + 8ll * (0) + 16ll * ((local_index / 4ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x464(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q13']	['q12']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4096ll) + 4096 + 8192ll * (0) + 16384ll * ((local_index / 4096ll) % 4ll);
		std::complex<double> t0 = temp[idx + 8192];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8192] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x465(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q6']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 64ll) + 128ll * ((local_index / 64ll) % 512ll);
		std::complex<double> t0 = temp[idx + 0 * 64];
		std::complex<double> t1 = temp[idx + 1 * 64];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 64] = std::complex<double>(t20, t21);
		temp[idx + 1 * 64] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x466(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q7']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 128ll) + 256ll * ((local_index / 128ll) % 256ll);
		std::complex<double> t0 = temp[idx + 0 * 128];
		std::complex<double> t1 = temp[idx + 1 * 128];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 128] = std::complex<double>(t20, t21);
		temp[idx + 1 * 128] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x467(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q10']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 1024ll) + 2048ll * ((local_index / 1024ll) % 32ll);
		std::complex<double> t0 = temp[idx + 0 * 1024];
		std::complex<double> t1 = temp[idx + 1 * 1024];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 1024] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1024] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x468(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q11']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2048ll) + 4096ll * ((local_index / 2048ll) % 16ll);
		std::complex<double> t0 = temp[idx + 2048];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2048] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x469(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q14']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 16384ll) + 32768ll * ((local_index / 16384ll) % 2ll);
		std::complex<double> t0 = temp[idx + 0 * 16384];
		std::complex<double> t1 = temp[idx + 1 * 16384];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 16384] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16384] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x470(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q15']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 32768ll) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 0 * 32768];
		std::complex<double> t1 = temp[idx + 1 * 32768];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 32768] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32768] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x471(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q5']	['q4']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 16 + 32ll * (0) + 64ll * ((local_index / 16ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x472(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 8192ll);
		std::complex<double> t0 = temp[idx + 0 * 4];
		std::complex<double> t1 = temp[idx + 1 * 4];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 4] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x473(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * ((local_index / 8ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 0 * 8];
		std::complex<double> t1 = temp[idx + 1 * 8];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 8] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x474(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q8']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 512ll * ((local_index / 256ll) % 128ll);
		std::complex<double> t0 = temp[idx + 256];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 256] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x475(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q9']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 512ll) + 1024ll * ((local_index / 512ll) % 64ll);
		std::complex<double> t0 = temp[idx + 0 * 512];
		std::complex<double> t1 = temp[idx + 1 * 512];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 512] = std::complex<double>(t20, t21);
		temp[idx + 1 * 512] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x476(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q1']	['q0']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (0) + 1 + 2ll * (0) + 4ll * (local_index % 16384ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x477(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (0) + 2ll * (local_index % 32768ll);
		std::complex<double> t0 = temp[idx + 0 * 1];
		std::complex<double> t1 = temp[idx + 1 * 1];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 1] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x478(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 16384ll);
		std::complex<double> t0 = temp[idx + 0 * 2];
		std::complex<double> t1 = temp[idx + 1 * 2];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 2] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x479(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q12']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4096ll) + 8192ll * ((local_index / 4096ll) % 8ll);
		std::complex<double> t0 = temp[idx + 4096];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4096] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x480(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q13']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 8192ll) + 16384ll * ((local_index / 8192ll) % 4ll);
		std::complex<double> t0 = temp[idx + 8192];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8192] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x481(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q7']	['q6']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 64ll) + 64 + 128ll * (0) + 256ll * ((local_index / 64ll) % 256ll);
		std::complex<double> t0 = temp[idx + 128];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 128] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x482(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q11']	['q10']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 1024ll) + 1024 + 2048ll * (0) + 4096ll * ((local_index / 1024ll) % 16ll);
		std::complex<double> t0 = temp[idx + 2048];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2048] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x483(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q15']	['q14']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 16384 + 32768ll * (0) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 32768];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32768] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x484(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q4']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 2048ll);
		std::complex<double> t0 = temp[idx + 0 * 16];
		std::complex<double> t1 = temp[idx + 1 * 16];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 16] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x485(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q5']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 32ll) + 64ll * ((local_index / 32ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 0 * 32];
		std::complex<double> t1 = temp[idx + 1 * 32];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 32] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x486(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q3']	['q2']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 4 + 8ll * (0) + 16ll * ((local_index / 4ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x487(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q9']	['q8']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 256 + 512ll * (0) + 1024ll * ((local_index / 256ll) % 64ll);
		std::complex<double> t0 = temp[idx + 512];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 512] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x488(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q5']	['q4']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 16 + 32ll * (0) + 64ll * ((local_index / 16ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x489(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 8192ll);
		std::complex<double> t0 = temp[idx + 0 * 4];
		std::complex<double> t1 = temp[idx + 1 * 4];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 4] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x490(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * ((local_index / 8ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 0 * 8];
		std::complex<double> t1 = temp[idx + 1 * 8];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 8] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x491(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q8']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 256ll) + 512ll * ((local_index / 256ll) % 128ll);
		std::complex<double> t0 = temp[idx + 0 * 256];
		std::complex<double> t1 = temp[idx + 1 * 256];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 256] = std::complex<double>(t20, t21);
		temp[idx + 1 * 256] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x492(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q9']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 512ll) + 1024ll * ((local_index / 512ll) % 64ll);
		std::complex<double> t0 = temp[idx + 0 * 512];
		std::complex<double> t1 = temp[idx + 1 * 512];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 512] = std::complex<double>(t20, t21);
		temp[idx + 1 * 512] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x493(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q1']	['q0']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (0) + 1 + 2ll * (0) + 4ll * (local_index % 16384ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x494(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q13']	['q12']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4096ll) + 4096 + 8192ll * (0) + 16384ll * ((local_index / 4096ll) % 4ll);
		std::complex<double> t0 = temp[idx + 8192];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8192] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x495(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q6']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 64ll) + 128ll * ((local_index / 64ll) % 512ll);
		std::complex<double> t0 = temp[idx + 64];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 64] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x496(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q7']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 128ll) + 256ll * ((local_index / 128ll) % 256ll);
		std::complex<double> t0 = temp[idx + 0 * 128];
		std::complex<double> t1 = temp[idx + 1 * 128];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 128] = std::complex<double>(t20, t21);
		temp[idx + 1 * 128] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x497(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q10']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 1024ll) + 2048ll * ((local_index / 1024ll) % 32ll);
		std::complex<double> t0 = temp[idx + 0 * 1024];
		std::complex<double> t1 = temp[idx + 1 * 1024];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 1024] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1024] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x498(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q11']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 2048ll) + 4096ll * ((local_index / 2048ll) % 16ll);
		std::complex<double> t0 = temp[idx + 0 * 2048];
		std::complex<double> t1 = temp[idx + 1 * 2048];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 2048] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2048] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x499(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q14']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 32768ll * ((local_index / 16384ll) % 2ll);
		std::complex<double> t0 = temp[idx + 16384];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 16384] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x500(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q15']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 32768ll) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 0 * 32768];
		std::complex<double> t1 = temp[idx + 1 * 32768];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 32768] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32768] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x501(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (0) + 2ll * (local_index % 32768ll);
		std::complex<double> t0 = temp[idx + 1];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 1] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x502(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 16384ll);
		std::complex<double> t0 = temp[idx + 0 * 2];
		std::complex<double> t1 = temp[idx + 1 * 2];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 2] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x503(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q12']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 4096ll) + 8192ll * ((local_index / 4096ll) % 8ll);
		std::complex<double> t0 = temp[idx + 0 * 4096];
		std::complex<double> t1 = temp[idx + 1 * 4096];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 4096] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4096] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x504(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q13']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 8192ll) + 16384ll * ((local_index / 8192ll) % 4ll);
		std::complex<double> t0 = temp[idx + 0 * 8192];
		std::complex<double> t1 = temp[idx + 1 * 8192];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 8192] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8192] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x505(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q7']	['q6']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 64ll) + 64 + 128ll * (0) + 256ll * ((local_index / 64ll) % 256ll);
		std::complex<double> t0 = temp[idx + 128];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 128] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x506(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q11']	['q10']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 1024ll) + 1024 + 2048ll * (0) + 4096ll * ((local_index / 1024ll) % 16ll);
		std::complex<double> t0 = temp[idx + 2048];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2048] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x507(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q15']	['q14']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 16384 + 32768ll * (0) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 32768];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32768] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x508(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q4']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 2048ll);
		std::complex<double> t0 = temp[idx + 0 * 16];
		std::complex<double> t1 = temp[idx + 1 * 16];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 16] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x509(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q5']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 32ll) + 64ll * ((local_index / 32ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 0 * 32];
		std::complex<double> t1 = temp[idx + 1 * 32];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 32] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x510(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q3']	['q2']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 4 + 8ll * (0) + 16ll * ((local_index / 4ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x511(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q9']	['q8']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 256 + 512ll * (0) + 1024ll * ((local_index / 256ll) % 64ll);
		std::complex<double> t0 = temp[idx + 512];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 512] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x512(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q1']	['q0']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (0) + 1 + 2ll * (0) + 4ll * (local_index % 16384ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x513(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q13']	['q12']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4096ll) + 4096 + 8192ll * (0) + 16384ll * ((local_index / 4096ll) % 4ll);
		std::complex<double> t0 = temp[idx + 8192];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8192] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x514(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q6']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 64ll) + 128ll * ((local_index / 64ll) % 512ll);
		std::complex<double> t0 = temp[idx + 0 * 64];
		std::complex<double> t1 = temp[idx + 1 * 64];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 64] = std::complex<double>(t20, t21);
		temp[idx + 1 * 64] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x515(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q7']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 128ll) + 256ll * ((local_index / 128ll) % 256ll);
		std::complex<double> t0 = temp[idx + 0 * 128];
		std::complex<double> t1 = temp[idx + 1 * 128];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 128] = std::complex<double>(t20, t21);
		temp[idx + 1 * 128] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x516(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q10']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 1024ll) + 2048ll * ((local_index / 1024ll) % 32ll);
		std::complex<double> t0 = temp[idx + 1024];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 1024] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x517(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q11']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 2048ll) + 4096ll * ((local_index / 2048ll) % 16ll);
		std::complex<double> t0 = temp[idx + 0 * 2048];
		std::complex<double> t1 = temp[idx + 1 * 2048];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 2048] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2048] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x518(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q14']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 32768ll * ((local_index / 16384ll) % 2ll);
		std::complex<double> t0 = temp[idx + 16384];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 16384] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x519(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q15']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 32768ll) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 32768];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32768] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x520(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q5']	['q4']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 16 + 32ll * (0) + 64ll * ((local_index / 16ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x521(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 8192ll);
		std::complex<double> t0 = temp[idx + 0 * 4];
		std::complex<double> t1 = temp[idx + 1 * 4];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 4] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x522(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * ((local_index / 8ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 0 * 8];
		std::complex<double> t1 = temp[idx + 1 * 8];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 8] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x523(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q8']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 256ll) + 512ll * ((local_index / 256ll) % 128ll);
		std::complex<double> t0 = temp[idx + 0 * 256];
		std::complex<double> t1 = temp[idx + 1 * 256];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 256] = std::complex<double>(t20, t21);
		temp[idx + 1 * 256] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x524(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q9']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 512ll) + 1024ll * ((local_index / 512ll) % 64ll);
		std::complex<double> t0 = temp[idx + 0 * 512];
		std::complex<double> t1 = temp[idx + 1 * 512];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 512] = std::complex<double>(t20, t21);
		temp[idx + 1 * 512] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x525(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q13']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 8192ll) + 16384ll * ((local_index / 8192ll) % 4ll);
		std::complex<double> t0 = temp[idx + 0 * 8192];
		std::complex<double> t1 = temp[idx + 1 * 8192];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 8192] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8192] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x526(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q7']	['q6']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 64ll) + 64 + 128ll * (0) + 256ll * ((local_index / 64ll) % 256ll);
		std::complex<double> t0 = temp[idx + 128];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 128] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x527(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q11']	['q10']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 1024ll) + 1024 + 2048ll * (0) + 4096ll * ((local_index / 1024ll) % 16ll);
		std::complex<double> t0 = temp[idx + 2048];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2048] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x528(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q15']	['q14']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 16384 + 32768ll * (0) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 32768];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32768] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x529(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q4']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 2048ll);
		std::complex<double> t0 = temp[idx + 0 * 16];
		std::complex<double> t1 = temp[idx + 1 * 16];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 16] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x530(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q5']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 32ll) + 64ll * ((local_index / 32ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 0 * 32];
		std::complex<double> t1 = temp[idx + 1 * 32];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 32] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x531(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q3']	['q2']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 4 + 8ll * (0) + 16ll * ((local_index / 4ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x532(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q9']	['q8']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 256 + 512ll * (0) + 1024ll * ((local_index / 256ll) % 64ll);
		std::complex<double> t0 = temp[idx + 512];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 512] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x533(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (0) + 2ll * (local_index % 32768ll);
		std::complex<double> t0 = temp[idx + 0 * 1];
		std::complex<double> t1 = temp[idx + 1 * 1];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 1] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x534(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 16384ll);
		std::complex<double> t0 = temp[idx + 0 * 2];
		std::complex<double> t1 = temp[idx + 1 * 2];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 2] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x535(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q12']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4096ll) + 8192ll * ((local_index / 4096ll) % 8ll);
		std::complex<double> t0 = temp[idx + 4096];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4096] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x536(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q1']	['q0']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (0) + 1 + 2ll * (0) + 4ll * (local_index % 16384ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x537(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q13']	['q12']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4096ll) + 4096 + 8192ll * (0) + 16384ll * ((local_index / 4096ll) % 4ll);
		std::complex<double> t0 = temp[idx + 8192];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8192] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x538(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q6']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 64ll) + 128ll * ((local_index / 64ll) % 512ll);
		std::complex<double> t0 = temp[idx + 0 * 64];
		std::complex<double> t1 = temp[idx + 1 * 64];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 64] = std::complex<double>(t20, t21);
		temp[idx + 1 * 64] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x539(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q7']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 128ll) + 256ll * ((local_index / 128ll) % 256ll);
		std::complex<double> t0 = temp[idx + 0 * 128];
		std::complex<double> t1 = temp[idx + 1 * 128];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 128] = std::complex<double>(t20, t21);
		temp[idx + 1 * 128] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x540(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q10']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 1024ll) + 2048ll * ((local_index / 1024ll) % 32ll);
		std::complex<double> t0 = temp[idx + 0 * 1024];
		std::complex<double> t1 = temp[idx + 1 * 1024];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 1024] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1024] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x541(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q11']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 2048ll) + 4096ll * ((local_index / 2048ll) % 16ll);
		std::complex<double> t0 = temp[idx + 0 * 2048];
		std::complex<double> t1 = temp[idx + 1 * 2048];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 2048] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2048] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x542(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q14']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 16384ll) + 32768ll * ((local_index / 16384ll) % 2ll);
		std::complex<double> t0 = temp[idx + 0 * 16384];
		std::complex<double> t1 = temp[idx + 1 * 16384];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 16384] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16384] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x543(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q15']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 32768ll) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 0 * 32768];
		std::complex<double> t1 = temp[idx + 1 * 32768];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 32768] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32768] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x544(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q5']	['q4']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 16 + 32ll * (0) + 64ll * ((local_index / 16ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x545(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 8192ll);
		std::complex<double> t0 = temp[idx + 0 * 4];
		std::complex<double> t1 = temp[idx + 1 * 4];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 4] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x546(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * ((local_index / 8ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 0 * 8];
		std::complex<double> t1 = temp[idx + 1 * 8];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 8] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x547(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q8']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 256ll) + 512ll * ((local_index / 256ll) % 128ll);
		std::complex<double> t0 = temp[idx + 0 * 256];
		std::complex<double> t1 = temp[idx + 1 * 256];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 256] = std::complex<double>(t20, t21);
		temp[idx + 1 * 256] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x548(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q9']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 512ll) + 1024ll * ((local_index / 512ll) % 64ll);
		std::complex<double> t0 = temp[idx + 0 * 512];
		std::complex<double> t1 = temp[idx + 1 * 512];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 512] = std::complex<double>(t20, t21);
		temp[idx + 1 * 512] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x549(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q9']	['q8']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 256 + 512ll * (0) + 1024ll * ((local_index / 256ll) % 64ll);
		std::complex<double> t0 = temp[idx + 512];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 512] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x550(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (0) + 2ll * (local_index % 32768ll);
		std::complex<double> t0 = temp[idx + 0 * 1];
		std::complex<double> t1 = temp[idx + 1 * 1];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 1] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x551(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 16384ll);
		std::complex<double> t0 = temp[idx + 0 * 2];
		std::complex<double> t1 = temp[idx + 1 * 2];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 2] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x552(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q12']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 4096ll) + 8192ll * ((local_index / 4096ll) % 8ll);
		std::complex<double> t0 = temp[idx + 0 * 4096];
		std::complex<double> t1 = temp[idx + 1 * 4096];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 4096] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4096] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x553(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q13']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 8192ll) + 16384ll * ((local_index / 8192ll) % 4ll);
		std::complex<double> t0 = temp[idx + 0 * 8192];
		std::complex<double> t1 = temp[idx + 1 * 8192];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 8192] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8192] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x554(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q7']	['q6']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 64ll) + 64 + 128ll * (0) + 256ll * ((local_index / 64ll) % 256ll);
		std::complex<double> t0 = temp[idx + 128];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 128] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x555(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q11']	['q10']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 1024ll) + 1024 + 2048ll * (0) + 4096ll * ((local_index / 1024ll) % 16ll);
		std::complex<double> t0 = temp[idx + 2048];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2048] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x556(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q15']	['q14']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 16384 + 32768ll * (0) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 32768];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32768] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x557(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q4']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 2048ll);
		std::complex<double> t0 = temp[idx + 16];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 16] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x558(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q5']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 32ll) + 64ll * ((local_index / 32ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 0 * 32];
		std::complex<double> t1 = temp[idx + 1 * 32];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 32] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x559(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q3']	['q2']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 4 + 8ll * (0) + 16ll * ((local_index / 4ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x560(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q13']	['q12']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4096ll) + 4096 + 8192ll * (0) + 16384ll * ((local_index / 4096ll) % 4ll);
		std::complex<double> t0 = temp[idx + 8192];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8192] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x561(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q6']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 64ll) + 128ll * ((local_index / 64ll) % 512ll);
		std::complex<double> t0 = temp[idx + 0 * 64];
		std::complex<double> t1 = temp[idx + 1 * 64];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 64] = std::complex<double>(t20, t21);
		temp[idx + 1 * 64] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x562(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q7']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 128ll) + 256ll * ((local_index / 128ll) % 256ll);
		std::complex<double> t0 = temp[idx + 0 * 128];
		std::complex<double> t1 = temp[idx + 1 * 128];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 128] = std::complex<double>(t20, t21);
		temp[idx + 1 * 128] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x563(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q10']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 1024ll) + 2048ll * ((local_index / 1024ll) % 32ll);
		std::complex<double> t0 = temp[idx + 0 * 1024];
		std::complex<double> t1 = temp[idx + 1 * 1024];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 1024] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1024] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x564(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q11']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 2048ll) + 4096ll * ((local_index / 2048ll) % 16ll);
		std::complex<double> t0 = temp[idx + 0 * 2048];
		std::complex<double> t1 = temp[idx + 1 * 2048];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 2048] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2048] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x565(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q14']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 16384ll) + 32768ll * ((local_index / 16384ll) % 2ll);
		std::complex<double> t0 = temp[idx + 0 * 16384];
		std::complex<double> t1 = temp[idx + 1 * 16384];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 16384] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16384] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x566(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q15']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 32768ll) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 0 * 32768];
		std::complex<double> t1 = temp[idx + 1 * 32768];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 32768] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32768] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x567(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q5']	['q4']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 16 + 32ll * (0) + 64ll * ((local_index / 16ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x568(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 8192ll);
		std::complex<double> t0 = temp[idx + 0 * 4];
		std::complex<double> t1 = temp[idx + 1 * 4];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 4] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x569(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 8ll) + 16ll * ((local_index / 8ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 0 * 8];
		std::complex<double> t1 = temp[idx + 1 * 8];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 8] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x570(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q8']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 256ll) + 512ll * ((local_index / 256ll) % 128ll);
		std::complex<double> t0 = temp[idx + 0 * 256];
		std::complex<double> t1 = temp[idx + 1 * 256];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 256] = std::complex<double>(t20, t21);
		temp[idx + 1 * 256] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x571(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q9']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 512ll) + 1024ll * ((local_index / 512ll) % 64ll);
		std::complex<double> t0 = temp[idx + 0 * 512];
		std::complex<double> t1 = temp[idx + 1 * 512];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 512] = std::complex<double>(t20, t21);
		temp[idx + 1 * 512] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x572(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q1']	['q0']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (0) + 1 + 2ll * (0) + 4ll * (local_index % 16384ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x573(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		size_t idx = 1ll * (0) + 2ll * (local_index % 32768ll);
		std::complex<double> t0 = temp[idx + 1];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 1] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x574(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 16384ll);
		std::complex<double> t0 = temp[idx + 0 * 2];
		std::complex<double> t1 = temp[idx + 1 * 2];
		double t20 = value.real() * t0.real() - value.imag() * t1.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t1.real();
		double t30 = value.real() * t1.real() - value.imag() * t0.imag();
		double t31 = value.real() * t1.imag() + value.imag() * t0.real();
		temp[idx + 0 * 2] = std::complex<double>(t20, t21);
		temp[idx + 1 * 2] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x575(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q12']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 4096ll) + 8192ll * ((local_index / 4096ll) % 8ll);
		std::complex<double> t0 = temp[idx + 0 * 4096];
		std::complex<double> t1 = temp[idx + 1 * 4096];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 4096] = std::complex<double>(t20, t21);
		temp[idx + 1 * 4096] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x576(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q13']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 8192ll) + 16384ll * ((local_index / 8192ll) % 4ll);
		std::complex<double> t0 = temp[idx + 0 * 8192];
		std::complex<double> t1 = temp[idx + 1 * 8192];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 8192] = std::complex<double>(t20, t21);
		temp[idx + 1 * 8192] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x577(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q7']	['q6']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 64ll) + 64 + 128ll * (0) + 256ll * ((local_index / 64ll) % 256ll);
		std::complex<double> t0 = temp[idx + 128];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 128] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x578(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q11']	['q10']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 1024ll) + 1024 + 2048ll * (0) + 4096ll * ((local_index / 1024ll) % 16ll);
		std::complex<double> t0 = temp[idx + 2048];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2048] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x579(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q15']	['q14']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16384ll) + 16384 + 32768ll * (0) + 65536ll * (0);
		std::complex<double> t0 = temp[idx + 32768];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32768] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x580(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q4']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 2048ll);
		std::complex<double> t0 = temp[idx + 0 * 16];
		std::complex<double> t1 = temp[idx + 1 * 16];
		double t20 = value.real() * t1.real() - value.imag() * t1.imag();
		double t21 = value.real() * t1.imag() + value.imag() * t1.real();
		double t30 = value.real() * t0.real() + value.imag() * t0.imag();
		double t31 = value.real() * t0.imag() - value.imag() * t0.real();
		temp[idx + 0 * 16] = std::complex<double>(t20, t21);
		temp[idx + 1 * 16] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x581(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q5']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 32ll) + 64ll * ((local_index / 32ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 0 * 32];
		std::complex<double> t1 = temp[idx + 1 * 32];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 32] = std::complex<double>(t20, t21);
		temp[idx + 1 * 32] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x582(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q3']	['q2']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 4 + 8ll * (0) + 16ll * ((local_index / 4ll) % 4096ll);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x583(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q9']	['q8']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 256ll) + 256 + 512ll * (0) + 1024ll * ((local_index / 256ll) % 64ll);
		std::complex<double> t0 = temp[idx + 512];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 512] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x584(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q1']	['q0']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (0) + 1 + 2ll * (0) + 4ll * (local_index % 16384ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x585(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q13']	['q12']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4096ll) + 4096 + 8192ll * (0) + 16384ll * ((local_index / 4096ll) % 4ll);
		std::complex<double> t0 = temp[idx + 8192];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8192] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x586(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q7']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 128ll) + 256ll * ((local_index / 128ll) % 256ll);
		std::complex<double> t0 = temp[idx + 0 * 128];
		std::complex<double> t1 = temp[idx + 1 * 128];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 128] = std::complex<double>(t20, t21);
		temp[idx + 1 * 128] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x587(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q10']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 1024ll) + 2048ll * ((local_index / 1024ll) % 32ll);
		std::complex<double> t0 = temp[idx + 0 * 1024];
		std::complex<double> t1 = temp[idx + 1 * 1024];
		double t20 = value.real() * t0.real() - value.imag() * t0.imag();
		double t21 = value.real() * t0.imag() + value.imag() * t0.real();
		double t30 = value.real() * t1.real() + value.imag() * t1.imag();
		double t31 = value.real() * t1.imag() - value.imag() * t1.real();
		temp[idx + 0 * 1024] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1024] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x588(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q5']	['q4']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 16384ll; local_index += 1){
		size_t idx = 1ll * (local_index % 16ll) + 16 + 32ll * (0) + 64ll * ((local_index / 16ll) % 1024ll);
		std::complex<double> t0 = temp[idx + 32];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 32] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x589(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q8']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 256ll) + 512ll * ((local_index / 256ll) % 128ll);
		std::complex<double> t0 = temp[idx + 0 * 256];
		std::complex<double> t1 = temp[idx + 1 * 256];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 256] = std::complex<double>(t20, t21);
		temp[idx + 1 * 256] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void local_s00001_apply_l2_c0x0x590(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q10']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 32768ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 1024ll) + 2048ll * ((local_index / 1024ll) % 32ll);
		std::complex<double> t0 = temp[idx + 0 * 1024];
		std::complex<double> t1 = temp[idx + 1 * 1024];
		double t20 = value * (t0.real() + t1.real());
		double t21 = value * (t0.imag() + t1.imag());
		double t30 = value * (t0.real() - t1.real());
		double t31 = value * (t0.imag() - t1.imag());
		temp[idx + 0 * 1024] = std::complex<double>(t20, t21);
		temp[idx + 1 * 1024] = std::complex<double>(t30, t31);
	}
}

inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x0(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x0(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x1(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x1(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x2(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x2(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x3(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x3(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x4(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x4(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x5(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x5(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x6(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x6(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x7(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x7(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x8(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x8(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x9(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x9(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x10(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x10(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x11(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x11(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x12(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x12(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x13(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x13(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x14(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x14(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x15(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x15(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x16(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x16(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x17(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x17(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x18(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x18(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x19(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x19(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x20(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x20(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x21(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x21(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x22(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x22(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x23(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x23(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x24(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x24(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x25(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x25(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x26(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x26(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x27(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x27(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x28(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x28(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x29(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x29(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x30(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x30(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x31(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x31(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x32(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x32(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x33(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x33(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x34(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x34(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x35(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x35(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x36(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x36(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x37(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x37(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x38(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x38(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x39(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x39(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x40(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x40(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x41(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x41(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x42(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x42(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x43(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x43(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x44(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x44(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x45(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x45(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x46(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x46(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x47(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x47(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x48(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x48(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x49(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x49(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x50(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x50(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x51(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x51(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x52(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x52(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x53(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x53(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x54(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x54(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x55(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x55(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x56(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x56(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x57(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x57(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x58(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x58(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x59(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x59(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x60(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x60(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x61(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x61(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x62(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x62(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x63(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x63(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x64(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x64(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x65(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x65(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x66(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x66(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x67(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x67(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x68(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x68(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x69(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x69(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x70(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x70(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x71(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x71(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x72(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x72(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x73(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x73(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x74(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x74(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x75(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x75(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x76(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x76(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x77(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x77(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x78(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x78(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x79(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x79(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x80(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x80(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x81(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x81(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x82(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x82(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x83(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x83(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x84(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x84(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x85(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x85(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x86(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x86(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x87(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x87(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x88(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x88(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x89(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x89(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x90(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x90(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x91(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x91(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x92(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x92(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x93(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x93(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x94(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x94(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x95(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x95(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x96(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x96(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x97(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x97(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x98(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x98(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x99(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x99(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x100(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x100(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x101(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x101(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x102(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x102(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x103(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x103(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x104(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x104(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x105(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x105(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x106(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x106(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x107(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x107(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x108(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x108(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x109(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x109(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x110(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x110(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x111(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x111(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x112(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x112(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x113(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x113(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x114(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x114(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x115(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x115(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x116(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x116(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x117(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x117(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x118(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x118(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x119(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x119(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x120(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x120(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x121(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x121(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x122(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x122(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x123(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x123(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x124(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x124(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x125(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x125(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x126(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x126(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x127(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x127(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x128(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x128(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x129(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x129(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x130(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x130(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x131(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x131(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x132(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x132(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x133(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x133(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x134(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x134(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x135(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x135(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x136(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x136(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x137(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x137(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x138(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x138(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x139(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x139(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x140(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x140(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x141(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x141(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x142(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x142(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x143(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x143(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x144(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x144(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x145(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x145(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x146(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x146(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x147(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x147(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x148(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x148(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x149(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x149(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x150(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x150(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x151(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x151(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x152(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x152(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x153(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x153(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x154(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x154(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x155(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x155(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x156(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x156(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x157(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x157(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x158(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x158(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x159(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x159(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x160(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x160(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x161(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x161(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x162(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x162(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x163(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x163(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x164(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x164(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x165(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x165(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x166(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x166(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x167(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x167(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x168(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x168(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x169(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x169(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x170(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x170(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x171(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x171(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x172(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x172(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x173(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x173(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x174(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x174(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x175(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x175(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x176(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x176(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x177(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x177(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x178(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x178(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x179(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x179(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x180(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x180(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x181(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x181(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x182(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x182(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x183(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x183(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x184(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x184(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x185(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x185(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x186(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x186(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x187(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x187(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x188(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x188(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x189(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x189(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x190(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x190(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x191(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x191(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x192(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x192(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x193(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x193(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x194(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x194(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x195(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x195(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x196(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x196(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x197(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x197(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x198(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x198(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x199(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x199(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x200(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x200(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x201(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x201(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x202(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x202(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x203(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x203(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x204(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x204(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x205(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x205(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x206(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x206(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x207(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x207(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x208(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x208(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x209(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x209(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x210(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x210(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x211(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x211(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x212(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x212(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x213(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x213(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x214(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x214(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x215(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x215(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x216(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x216(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x217(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x217(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x218(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x218(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x219(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x219(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x220(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x220(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x221(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x221(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x222(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x222(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x223(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x223(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x224(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x224(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x225(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x225(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x226(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x226(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x227(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x227(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x228(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x228(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x229(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x229(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x230(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x230(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x231(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x231(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x232(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x232(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x233(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x233(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x234(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x234(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x235(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x235(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x236(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x236(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x237(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x237(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x238(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x238(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x239(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x239(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x240(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x240(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x241(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x241(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x242(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x242(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x243(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x243(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x244(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x244(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x245(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x245(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x246(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x246(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x247(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x247(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x248(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x248(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x249(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x249(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x250(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x250(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x251(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x251(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x252(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x252(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x253(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x253(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x254(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x254(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x255(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x255(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x256(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x256(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x257(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x257(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x258(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x258(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x259(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x259(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x260(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x260(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x261(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x261(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x262(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x262(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x263(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x263(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x264(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x264(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x265(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x265(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x266(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x266(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x267(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x267(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x268(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x268(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x269(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x269(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x270(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x270(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x271(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x271(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x272(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x272(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x273(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x273(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x274(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x274(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x275(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x275(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x276(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x276(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x277(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x277(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x278(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x278(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x279(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x279(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x280(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x280(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x281(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x281(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x282(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x282(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x283(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x283(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x284(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x284(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x285(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x285(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x286(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x286(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x287(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x287(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x288(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x288(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x289(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x289(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x290(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x290(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x291(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x291(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x292(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x292(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x293(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x293(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x294(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x294(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x295(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x295(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x296(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x296(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x297(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x297(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x298(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x298(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x299(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x299(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x300(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x300(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x301(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x301(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x302(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x302(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x303(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x303(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x304(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x304(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x305(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x305(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x306(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x306(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x307(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x307(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x308(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x308(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x309(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x309(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x310(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x310(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x311(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x311(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x312(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x312(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x313(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x313(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x314(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x314(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x315(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x315(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x316(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x316(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x317(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x317(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x318(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x318(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x319(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x319(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x320(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x320(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x321(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x321(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x322(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x322(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x323(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x323(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x324(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x324(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x325(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x325(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x326(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x326(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x327(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x327(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x328(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x328(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x329(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x329(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x330(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x330(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x331(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x331(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x332(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x332(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x333(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x333(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x334(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x334(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x335(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x335(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x336(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x336(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x337(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x337(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x338(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x338(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x339(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x339(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x340(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x340(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x341(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x341(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x342(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x342(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x343(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x343(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x344(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x344(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x345(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x345(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x346(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x346(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x347(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x347(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x348(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x348(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x349(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x349(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x350(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x350(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x351(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x351(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x352(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x352(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x353(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x353(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x354(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x354(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x355(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x355(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x356(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x356(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x357(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x357(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x358(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x358(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x359(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x359(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x360(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x360(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x361(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x361(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x362(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x362(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x363(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x363(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x364(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x364(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x365(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x365(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x366(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x366(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x367(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x367(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x368(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x368(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x369(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x369(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x370(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x370(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x371(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x371(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x372(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x372(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x373(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x373(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x374(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x374(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x375(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x375(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x376(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x376(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x377(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x377(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x378(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x378(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x379(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x379(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x380(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x380(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x381(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x381(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x382(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x382(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x383(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x383(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x384(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x384(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x385(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x385(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x386(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x386(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x387(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x387(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x388(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x388(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x389(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x389(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x390(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x390(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x391(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x391(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x392(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x392(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x393(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x393(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x394(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x394(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x395(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x395(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x396(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x396(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x397(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x397(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x398(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x398(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x399(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x399(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x400(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x400(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x401(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x401(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x402(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x402(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x403(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x403(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x404(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x404(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x405(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x405(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x406(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x406(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x407(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x407(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x408(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x408(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x409(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x409(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x410(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x410(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x411(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x411(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x412(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x412(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x413(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x413(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x414(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x414(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x415(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x415(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x416(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x416(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x417(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x417(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x418(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x418(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x419(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x419(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x420(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x420(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x421(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x421(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x422(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x422(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x423(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x423(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x424(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x424(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x425(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x425(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x426(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x426(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x427(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x427(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x428(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x428(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x429(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x429(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x430(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x430(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x431(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x431(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x432(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x432(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x433(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x433(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x434(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x434(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x435(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x435(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x436(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x436(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x437(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x437(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x438(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x438(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x439(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x439(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x440(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x440(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x441(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x441(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x442(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x442(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x443(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x443(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x444(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x444(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x445(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x445(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x446(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x446(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x447(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x447(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x448(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x448(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x449(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x449(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x450(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x450(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x451(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x451(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x452(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x452(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x453(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x453(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x454(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x454(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x455(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x455(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x456(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x456(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x457(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x457(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x458(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x458(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x459(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x459(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x460(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x460(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x461(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x461(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x462(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x462(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x463(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x463(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x464(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x464(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x465(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x465(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x466(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x466(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x467(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x467(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x468(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x468(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x469(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x469(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x470(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x470(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x471(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x471(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x472(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x472(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x473(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x473(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x474(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x474(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x475(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x475(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x476(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x476(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x477(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x477(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x478(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x478(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x479(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x479(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x480(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x480(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x481(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x481(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x482(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x482(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x483(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x483(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x484(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x484(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x485(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x485(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x486(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x486(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x487(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x487(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x488(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x488(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x489(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x489(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x490(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x490(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x491(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x491(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x492(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x492(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x493(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x493(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x494(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x494(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x495(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x495(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x496(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x496(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x497(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x497(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x498(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x498(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x499(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x499(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x500(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x500(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x501(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x501(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x502(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x502(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x503(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x503(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x504(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x504(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x505(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x505(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x506(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x506(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x507(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x507(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x508(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x508(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x509(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x509(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x510(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x510(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x511(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x511(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x512(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x512(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x513(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x513(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x514(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x514(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x515(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x515(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x516(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x516(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x517(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x517(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x518(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x518(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x519(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x519(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x520(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x520(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x521(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x521(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x522(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x522(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x523(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x523(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x524(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x524(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x525(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x525(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x526(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x526(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x527(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x527(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x528(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x528(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x529(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x529(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x530(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x530(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x531(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x531(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x532(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x532(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x533(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x533(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x534(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x534(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x535(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x535(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x536(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x536(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x537(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x537(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x538(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x538(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x539(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x539(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x540(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x540(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x541(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x541(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x542(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x542(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x543(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x543(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x544(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x544(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x545(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x545(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x546(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x546(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x547(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x547(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x548(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x548(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x549(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x549(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x550(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x550(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x551(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x551(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x552(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x552(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x553(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x553(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x554(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x554(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x555(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x555(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x556(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x556(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x557(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x557(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x558(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x558(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x559(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x559(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x560(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x560(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x561(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x561(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x562(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x562(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x563(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x563(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x564(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x564(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x565(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x565(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x566(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x566(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x567(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x567(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x568(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x568(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x569(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x569(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x570(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x570(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x571(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x571(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x572(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x572(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x573(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x573(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x574(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x574(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x575(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x575(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x576(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x576(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x577(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x577(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x578(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x578(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x579(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x579(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x580(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x580(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x581(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x581(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x582(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x582(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x583(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x583(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x584(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x584(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x585(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x585(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x586(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x586(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x587(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x587(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x588(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x588(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x589(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x589(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00001_apply_l2_c0x0x590(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00001_apply_l2_c0x0x590(temp, constant_values, iter0);
}

void s00001_apply_l1_c0x0(std::complex<double> *io_temp, std::complex<double>* constant_values, size_t iter0){
	s00001_apply_l2_c0x0x0(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x1(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x2(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x3(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x4(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x5(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x6(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x7(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x8(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x9(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x10(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x11(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x12(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x13(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x14(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x15(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x16(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x17(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x18(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x19(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x20(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x21(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x22(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x23(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x24(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x25(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x26(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x27(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x28(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x29(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x30(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x31(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x32(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x33(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x34(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x35(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x36(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x37(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x38(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x39(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x40(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x41(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x42(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x43(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x44(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x45(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x46(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x47(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x48(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x49(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x50(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x51(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x52(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x53(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x54(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x55(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x56(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x57(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x58(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x59(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x60(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x61(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x62(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x63(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x64(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x65(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x66(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x67(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x68(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x69(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x70(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x71(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x72(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x73(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x74(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x75(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x76(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x77(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x78(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x79(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x80(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x81(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x82(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x83(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x84(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x85(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x86(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x87(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x88(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x89(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x90(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x91(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x92(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x93(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x94(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x95(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x96(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x97(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x98(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x99(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x100(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x101(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x102(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x103(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x104(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x105(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x106(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x107(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x108(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x109(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x110(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x111(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x112(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x113(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x114(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x115(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x116(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x117(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x118(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x119(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x120(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x121(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x122(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x123(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x124(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x125(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x126(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x127(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x128(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x129(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x130(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x131(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x132(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x133(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x134(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x135(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x136(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x137(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x138(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x139(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x140(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x141(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x142(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x143(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x144(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x145(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x146(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x147(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x148(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x149(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x150(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x151(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x152(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x153(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x154(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x155(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x156(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x157(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x158(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x159(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x160(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x161(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x162(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x163(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x164(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x165(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x166(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x167(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x168(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x169(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x170(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x171(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x172(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x173(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x174(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x175(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x176(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x177(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x178(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x179(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x180(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x181(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x182(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x183(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x184(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x185(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x186(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x187(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x188(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x189(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x190(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x191(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x192(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x193(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x194(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x195(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x196(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x197(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x198(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x199(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x200(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x201(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x202(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x203(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x204(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x205(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x206(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x207(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x208(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x209(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x210(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x211(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x212(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x213(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x214(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x215(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x216(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x217(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x218(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x219(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x220(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x221(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x222(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x223(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x224(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x225(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x226(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x227(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x228(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x229(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x230(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x231(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x232(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x233(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x234(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x235(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x236(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x237(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x238(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x239(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x240(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x241(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x242(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x243(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x244(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x245(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x246(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x247(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x248(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x249(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x250(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x251(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x252(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x253(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x254(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x255(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x256(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x257(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x258(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x259(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x260(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x261(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x262(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x263(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x264(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x265(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x266(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x267(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x268(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x269(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x270(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x271(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x272(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x273(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x274(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x275(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x276(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x277(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x278(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x279(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x280(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x281(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x282(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x283(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x284(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x285(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x286(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x287(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x288(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x289(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x290(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x291(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x292(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x293(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x294(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x295(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x296(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x297(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x298(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x299(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x300(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x301(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x302(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x303(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x304(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x305(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x306(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x307(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x308(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x309(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x310(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x311(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x312(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x313(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x314(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x315(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x316(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x317(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x318(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x319(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x320(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x321(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x322(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x323(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x324(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x325(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x326(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x327(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x328(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x329(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x330(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x331(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x332(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x333(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x334(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x335(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x336(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x337(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x338(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x339(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x340(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x341(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x342(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x343(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x344(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x345(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x346(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x347(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x348(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x349(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x350(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x351(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x352(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x353(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x354(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x355(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x356(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x357(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x358(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x359(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x360(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x361(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x362(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x363(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x364(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x365(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x366(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x367(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x368(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x369(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x370(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x371(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x372(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x373(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x374(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x375(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x376(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x377(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x378(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x379(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x380(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x381(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x382(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x383(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x384(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x385(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x386(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x387(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x388(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x389(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x390(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x391(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x392(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x393(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x394(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x395(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x396(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x397(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x398(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x399(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x400(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x401(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x402(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x403(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x404(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x405(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x406(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x407(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x408(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x409(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x410(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x411(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x412(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x413(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x414(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x415(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x416(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x417(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x418(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x419(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x420(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x421(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x422(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x423(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x424(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x425(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x426(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x427(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x428(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x429(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x430(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x431(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x432(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x433(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x434(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x435(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x436(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x437(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x438(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x439(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x440(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x441(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x442(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x443(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x444(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x445(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x446(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x447(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x448(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x449(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x450(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x451(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x452(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x453(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x454(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x455(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x456(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x457(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x458(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x459(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x460(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x461(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x462(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x463(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x464(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x465(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x466(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x467(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x468(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x469(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x470(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x471(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x472(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x473(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x474(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x475(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x476(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x477(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x478(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x479(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x480(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x481(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x482(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x483(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x484(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x485(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x486(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x487(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x488(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x489(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x490(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x491(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x492(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x493(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x494(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x495(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x496(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x497(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x498(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x499(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x500(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x501(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x502(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x503(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x504(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x505(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x506(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x507(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x508(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x509(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x510(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x511(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x512(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x513(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x514(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x515(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x516(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x517(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x518(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x519(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x520(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x521(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x522(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x523(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x524(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x525(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x526(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x527(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x528(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x529(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x530(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x531(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x532(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x533(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x534(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x535(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x536(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x537(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x538(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x539(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x540(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x541(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x542(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x543(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x544(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x545(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x546(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x547(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x548(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x549(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x550(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x551(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x552(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x553(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x554(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x555(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x556(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x557(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x558(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x559(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x560(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x561(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x562(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x563(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x564(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x565(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x566(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x567(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x568(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x569(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x570(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x571(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x572(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x573(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x574(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x575(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x576(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x577(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x578(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x579(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x580(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x581(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x582(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x583(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x584(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x585(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x586(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x587(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x588(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x589(io_temp, constant_values, iter0);
	s00001_apply_l2_c0x0x590(io_temp, constant_values, iter0);
}

