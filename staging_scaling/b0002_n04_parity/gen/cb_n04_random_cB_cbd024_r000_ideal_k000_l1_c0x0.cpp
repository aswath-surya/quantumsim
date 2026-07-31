#include <omp.h>

#include "cb_n04_random_cB_cbd024_r000_ideal_k000_l1_c0x0.hpp"

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x0(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (0) + 2ll * (local_index % 8ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x1(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 4ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x2(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 2ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x3(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * (0);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x4(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q2']	['q1']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 4ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 2 + 4ll * (0) + 8ll * ((local_index / 2ll) % 2ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x5(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (0) + 2ll * (local_index % 8ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x6(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * (0);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x7(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * (0);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x8(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 4ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x9(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 2ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x10(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		size_t idx = 1ll * (0) + 2ll * (local_index % 8ll);
		std::complex<double> t0 = temp[idx + 1];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 1] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x11(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (0) + 2ll * (local_index % 8ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x12(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q2']	['q1']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 4ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 2 + 4ll * (0) + 8ll * ((local_index / 2ll) % 2ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x13(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * (0);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x14(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 2ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x15(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (0) + 2ll * (local_index % 8ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x16(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		size_t idx = 1ll * (local_index % 8ll) + 16ll * (0);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x17(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 4ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x18(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * (0);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x19(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q2']	['q1']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 4ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 2 + 4ll * (0) + 8ll * ((local_index / 2ll) % 2ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x20(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (0) + 2ll * (local_index % 8ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x21(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		size_t idx = 1ll * (0) + 2ll * (local_index % 8ll);
		std::complex<double> t0 = temp[idx + 1];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 1] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x22(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * (0);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x23(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 4ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x24(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 2ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x25(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * (0);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x26(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (0) + 2ll * (local_index % 8ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x27(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q2']	['q1']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 4ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 2 + 4ll * (0) + 8ll * ((local_index / 2ll) % 2ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x28(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 4ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x29(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 2ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x30(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (0) + 2ll * (local_index % 8ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x31(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * (0);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x32(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (0) + 2ll * (local_index % 8ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x33(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q2']	['q1']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 4ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 2 + 4ll * (0) + 8ll * ((local_index / 2ll) % 2ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x34(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		size_t idx = 1ll * (local_index % 8ll) + 16ll * (0);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x35(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (0) + 2ll * (local_index % 8ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x36(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * (0);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x37(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 4ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x38(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 2ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x39(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q2']	['q1']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 4ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 2 + 4ll * (0) + 8ll * ((local_index / 2ll) % 2ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x40(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (0) + 2ll * (local_index % 8ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x41(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * (0);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x42(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (0) + 2ll * (local_index % 8ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x43(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * (0);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x44(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 4ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x45(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 2ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x46(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * (0);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x47(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (0) + 2ll * (local_index % 8ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x48(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q2']	['q1']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 4ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 2 + 4ll * (0) + 8ll * ((local_index / 2ll) % 2ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x49(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 4ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x50(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 2ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x51(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (0) + 2ll * (local_index % 8ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x52(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * (0);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x53(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * (0);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x54(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q2']	['q1']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 4ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 2 + 4ll * (0) + 8ll * ((local_index / 2ll) % 2ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x55(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (0) + 2ll * (local_index % 8ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x56(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (0) + 2ll * (local_index % 8ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x57(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 4ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x58(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 2ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x59(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		size_t idx = 1ll * (local_index % 8ll) + 16ll * (0);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x60(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q2']	['q1']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 4ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 2 + 4ll * (0) + 8ll * ((local_index / 2ll) % 2ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x61(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (0) + 2ll * (local_index % 8ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x62(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * (0);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x63(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		size_t idx = 1ll * (local_index % 8ll) + 16ll * (0);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x64(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 4ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x65(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 2ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x66(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (0) + 2ll * (local_index % 8ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x67(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		size_t idx = 1ll * (0) + 2ll * (local_index % 8ll);
		std::complex<double> t0 = temp[idx + 1];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 1] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x68(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q2']	['q1']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 4ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 2 + 4ll * (0) + 8ll * ((local_index / 2ll) % 2ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x69(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * (0);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x70(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 2ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x71(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		size_t idx = 1ll * (local_index % 8ll) + 16ll * (0);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x72(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (0) + 2ll * (local_index % 8ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x73(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 4ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x74(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		size_t idx = 1ll * (local_index % 8ll) + 16ll * (0);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x75(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q2']	['q1']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 4ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 2 + 4ll * (0) + 8ll * ((local_index / 2ll) % 2ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x76(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (0) + 2ll * (local_index % 8ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x77(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (0) + 2ll * (local_index % 8ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x78(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 8ll) + 16ll * (0);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x79(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 4ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x80(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 2ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x81(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q3']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		size_t idx = 1ll * (local_index % 8ll) + 16ll * (0);
		std::complex<double> t0 = temp[idx + 8];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 8] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x82(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (0) + 2ll * (local_index % 8ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x83(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q2']	['q1']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 4ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 2 + 4ll * (0) + 8ll * ((local_index / 2ll) % 2ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x84(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 4ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x85(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 2ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x86(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q2']	['q1']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 4ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 2 + 4ll * (0) + 8ll * ((local_index / 2ll) % 2ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x87(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 4ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x88(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 2ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x89(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q2']	['q1']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 4ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 2 + 4ll * (0) + 8ll * ((local_index / 2ll) % 2ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x90(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 4ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x91(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 2ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x92(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q2']	['q1']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 4ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 2 + 4ll * (0) + 8ll * ((local_index / 2ll) % 2ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x93(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 4ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x94(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 2ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x95(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q2']	['q1']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 4ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 2 + 4ll * (0) + 8ll * ((local_index / 2ll) % 2ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x96(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 4ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x97(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 2ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x98(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q2']	['q1']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 4ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 2 + 4ll * (0) + 8ll * ((local_index / 2ll) % 2ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x99(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 4ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x100(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 2ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x101(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q2']	['q1']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 4ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 2 + 4ll * (0) + 8ll * ((local_index / 2ll) % 2ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x102(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//x	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(1.0, 0.0);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 2ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x103(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 4ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x104(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q2']	['q1']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 4ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 2 + 4ll * (0) + 8ll * ((local_index / 2ll) % 2ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x105(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 4ll);
		std::complex<double> t0 = temp[idx + 2];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 2] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x106(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 2ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x107(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q2']	['q1']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 4ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 2 + 4ll * (0) + 8ll * ((local_index / 2ll) % 2ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x108(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 4ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x109(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//z	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 2ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x110(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q2']	['q1']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 4ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 2 + 4ll * (0) + 8ll * ((local_index / 2ll) % 2ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x111(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 4ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x112(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//y	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.0, -1.0);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 2ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x113(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q2']	['q1']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 4ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 2 + 4ll * (0) + 8ll * ((local_index / 2ll) % 2ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x114(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 4ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x115(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rx	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, -0.7071067811865475);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 2ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x116(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q2']	['q1']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 4ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 2 + 4ll * (0) + 8ll * ((local_index / 2ll) % 2ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x117(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 4ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x118(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 2ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x119(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//cz	['q2']	['q1']
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 4ll; local_index += 1){
		size_t idx = 1ll * (local_index % 2ll) + 2 + 4ll * (0) + 8ll * ((local_index / 2ll) % 2ll);
		std::complex<double> t0 = temp[idx + 4];
		double t10 = -1.0 * t0.real();
		double t11 = -1.0 * t0.imag();
		temp[idx + 4] = std::complex<double>(t10, t11);
	}
}

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x120(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q1']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 4ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x121(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//rz	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		std::complex<double> value = std::complex<double>(0.7071067811865475, 0.7071067811865475);
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 2ll);
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

inline __attribute__((always_inline)) void local_s00002_apply_l2_c0x0x122(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//h	['q2']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 8ll; local_index += 1){
		double value = 0.7071067811865475;
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 2ll);
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

inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x0(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x0(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x1(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x1(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x2(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x2(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x3(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x3(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x4(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x4(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x5(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x5(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x6(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x6(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x7(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x7(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x8(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x8(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x9(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x9(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x10(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x10(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x11(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x11(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x12(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x12(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x13(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x13(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x14(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x14(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x15(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x15(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x16(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x16(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x17(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x17(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x18(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x18(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x19(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x19(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x20(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x20(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x21(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x21(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x22(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x22(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x23(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x23(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x24(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x24(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x25(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x25(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x26(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x26(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x27(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x27(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x28(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x28(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x29(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x29(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x30(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x30(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x31(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x31(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x32(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x32(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x33(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x33(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x34(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x34(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x35(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x35(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x36(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x36(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x37(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x37(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x38(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x38(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x39(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x39(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x40(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x40(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x41(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x41(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x42(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x42(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x43(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x43(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x44(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x44(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x45(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x45(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x46(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x46(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x47(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x47(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x48(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x48(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x49(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x49(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x50(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x50(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x51(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x51(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x52(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x52(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x53(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x53(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x54(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x54(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x55(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x55(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x56(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x56(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x57(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x57(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x58(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x58(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x59(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x59(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x60(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x60(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x61(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x61(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x62(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x62(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x63(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x63(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x64(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x64(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x65(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x65(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x66(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x66(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x67(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x67(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x68(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x68(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x69(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x69(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x70(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x70(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x71(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x71(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x72(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x72(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x73(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x73(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x74(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x74(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x75(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x75(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x76(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x76(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x77(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x77(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x78(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x78(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x79(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x79(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x80(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x80(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x81(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x81(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x82(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x82(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x83(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x83(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x84(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x84(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x85(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x85(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x86(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x86(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x87(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x87(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x88(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x88(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x89(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x89(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x90(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x90(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x91(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x91(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x92(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x92(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x93(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x93(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x94(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x94(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x95(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x95(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x96(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x96(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x97(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x97(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x98(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x98(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x99(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x99(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x100(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x100(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x101(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x101(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x102(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x102(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x103(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x103(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x104(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x104(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x105(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x105(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x106(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x106(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x107(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x107(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x108(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x108(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x109(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x109(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x110(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x110(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x111(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x111(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x112(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x112(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x113(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x113(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x114(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x114(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x115(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x115(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x116(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x116(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x117(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x117(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x118(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x118(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x119(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x119(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x120(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x120(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x121(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x121(temp, constant_values, iter0);
}
inline __attribute__((always_inline)) void s00002_apply_l2_c0x0x122(std::complex<double> *temp, std::complex<double> *constant_values, size_t iter0) {
	local_s00002_apply_l2_c0x0x122(temp, constant_values, iter0);
}

void s00002_apply_l1_c0x0(std::complex<double> *io_temp, std::complex<double>* constant_values, size_t iter0){
	s00002_apply_l2_c0x0x0(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x1(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x2(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x3(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x4(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x5(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x6(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x7(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x8(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x9(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x10(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x11(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x12(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x13(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x14(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x15(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x16(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x17(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x18(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x19(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x20(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x21(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x22(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x23(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x24(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x25(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x26(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x27(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x28(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x29(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x30(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x31(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x32(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x33(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x34(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x35(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x36(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x37(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x38(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x39(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x40(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x41(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x42(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x43(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x44(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x45(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x46(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x47(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x48(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x49(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x50(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x51(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x52(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x53(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x54(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x55(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x56(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x57(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x58(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x59(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x60(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x61(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x62(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x63(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x64(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x65(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x66(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x67(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x68(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x69(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x70(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x71(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x72(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x73(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x74(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x75(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x76(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x77(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x78(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x79(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x80(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x81(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x82(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x83(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x84(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x85(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x86(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x87(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x88(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x89(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x90(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x91(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x92(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x93(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x94(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x95(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x96(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x97(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x98(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x99(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x100(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x101(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x102(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x103(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x104(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x105(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x106(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x107(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x108(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x109(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x110(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x111(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x112(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x113(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x114(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x115(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x116(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x117(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x118(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x119(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x120(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x121(io_temp, constant_values, iter0);
	s00002_apply_l2_c0x0x122(io_temp, constant_values, iter0);
}

