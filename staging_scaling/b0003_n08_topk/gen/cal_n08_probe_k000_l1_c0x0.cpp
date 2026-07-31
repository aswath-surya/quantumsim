#include <omp.h>

#include "cal_n08_probe_k000_l1_c0x0.hpp"

inline __attribute__((always_inline)) void local_s00000_apply_l2_c0x0x0(std::complex<double> *temp, std::complex<double>* constant_values, size_t iter0){
	//ry	['q0']	[]
#pragma omp parallel for schedule(static, 64)
	for(size_t local_index = 0; local_index < 128ll; local_index += 1){
		std::complex<double> value = constant_values[0];
		size_t idx = 1ll * (0) + 2ll * (local_index % 128ll);
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
	for(size_t local_index = 0; local_index < 128ll; local_index += 1){
		std::complex<double> value = constant_values[1];
		size_t idx = 1ll * (local_index % 2ll) + 4ll * ((local_index / 2ll) % 64ll);
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
	for(size_t local_index = 0; local_index < 128ll; local_index += 1){
		std::complex<double> value = constant_values[2];
		size_t idx = 1ll * (local_index % 4ll) + 8ll * ((local_index / 4ll) % 32ll);
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
	for(size_t local_index = 0; local_index < 128ll; local_index += 1){
		std::complex<double> value = constant_values[3];
		size_t idx = 1ll * (local_index % 8ll) + 16ll * ((local_index / 8ll) % 16ll);
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
	for(size_t local_index = 0; local_index < 128ll; local_index += 1){
		std::complex<double> value = constant_values[4];
		size_t idx = 1ll * (local_index % 16ll) + 32ll * ((local_index / 16ll) % 8ll);
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
	for(size_t local_index = 0; local_index < 128ll; local_index += 1){
		std::complex<double> value = constant_values[5];
		size_t idx = 1ll * (local_index % 32ll) + 64ll * ((local_index / 32ll) % 4ll);
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
	for(size_t local_index = 0; local_index < 128ll; local_index += 1){
		std::complex<double> value = constant_values[6];
		size_t idx = 1ll * (local_index % 64ll) + 128ll * ((local_index / 64ll) % 2ll);
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
	for(size_t local_index = 0; local_index < 128ll; local_index += 1){
		std::complex<double> value = constant_values[7];
		size_t idx = 1ll * (local_index % 128ll) + 256ll * (0);
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

void s00000_apply_l1_c0x0(std::complex<double> *io_temp, std::complex<double>* constant_values, size_t iter0){
	s00000_apply_l2_c0x0x0(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x1(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x2(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x3(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x4(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x5(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x6(io_temp, constant_values, iter0);
	s00000_apply_l2_c0x0x7(io_temp, constant_values, iter0);
}

