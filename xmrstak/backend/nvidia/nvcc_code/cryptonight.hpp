#pragma once

#include <stdint.h>
#include <string>

#include "xmrstak/backend/cryptonight.hpp"
#include "xmrstak/jconf.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

typedef struct
{
	int device_id;
	const char* device_name;
	int device_arch[2];
	int device_mpcount;
	int device_blocks;
	int device_threads;
	int device_bfactor;
	int device_bsleep;
	int syncMode;
	bool memMode;
	int cIdx;

	uint32_t* foundNonce[2];
	uint32_t* foundCount[2];
	cudaStream_t stream[2];

	uint32_t nonce[2];

	uint32_t* d_input;
	uint32_t inputlen;
	uint32_t* d_result_countv[2];
	uint32_t* d_result_noncev[2];
	uint32_t* d_long_statev[2];
	uint32_t* d_ctx_statev[2];
	uint32_t* d_ctx_state2v[2];
	uint32_t* d_ctx_av[2];
	uint32_t* d_ctx_bv[2];
	uint32_t* d_ctx_key1v[2];
	uint32_t* d_ctx_key2v[2];
	uint32_t* d_ctx_textv[2];
	std::string name;
	size_t free_device_memory;
	size_t total_device_memory;

	CUcontext cuContext;
	CUmodule module = nullptr;
	CUfunction kernel = nullptr;
	uint64_t kernel_height = 0;
	xmrstak_algo cached_algo = {xmrstak_algo_id::invalid_algo};
} nvid_ctx;

extern "C"
{

	/** get device count
 *
 * @param deviceCount[out] cuda device count
 * @return error code: 0 == error is occurred, 1 == no error
 */
	int cuda_get_devicecount(int* deviceCount);
	int cuda_get_deviceinfo(nvid_ctx* ctx);
	int cryptonight_extra_cpu_init(nvid_ctx* ctx);
	void cryptonight_sync(nvid_ctx* ctx, uint32_t startNonce, uint32_t* rescount, uint32_t* resnonce);
	void cryptonight_extra_cpu_set_data(nvid_ctx* ctx, const void* data, uint32_t len);
	void cryptonight_extra_cpu_prepare(nvid_ctx* ctx, uint32_t startNonce, const xmrstak_algo& miner_algo);
	void cryptonight_extra_cpu_final(nvid_ctx* ctx, uint32_t startNonce, uint64_t target, uint32_t* rescount, uint32_t* resnonce, const xmrstak_algo& miner_algo);
}

void cryptonight_core_cpu_hash(nvid_ctx* ctx, const xmrstak_algo& miner_algo, uint32_t startNonce, uint64_t chain_height);
