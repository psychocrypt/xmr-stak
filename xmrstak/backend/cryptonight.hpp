#pragma once
#include <stddef.h>
#include <inttypes.h>
#include <type_traits>

enum xmrstak_algo_id
{
	invalid_algo = 0,
	cryptonight = 1,
	cryptonight_lite = 2,
	cryptonight_monero = 3,
	cryptonight_heavy = 4,
	cryptonight_aeon = 5,
	cryptonight_ipbc = 6, // equal to cryptonight_aeon with a small tweak in the miner code
	cryptonight_stellite = 7, //equal to cryptonight_monero but with one tiny change
	cryptonight_masari = 8, //equal to cryptonight_monero but with less iterations, used by masari
	cryptonight_haven = 9, // equal to cryptonight_heavy with a small tweak
	cryptonight_bittube2 = 10, // derived from cryptonight_heavy with own aes-round implementation and minor other tweaks
	cryptonight_monero_v8 = 11,
	cryptonight_superfast = 12,
	cryptonight_gpu = 13,
	cryptonight_turtle = 14
};

inline uint32_t cn_mask_bytes(xmrstak_algo_id algo)
{
	switch(algo)
	{
	case cryptonight_gpu:
		return 64u;
	default:
		return 16u;
	}
}

struct xmrstak_algo
{
	xmrstak_algo(xmrstak_algo_id algorithm) : algo(algorithm)
	{
	}
	xmrstak_algo(xmrstak_algo_id algorithm, uint32_t iteration) : algo(algorithm), iter(iteration)
	{
	}
	xmrstak_algo(xmrstak_algo_id algorithm, uint32_t iteration, uint32_t memory) : algo(algorithm), iter(iteration), mem(memory)
	{
	}
	xmrstak_algo(xmrstak_algo_id algorithm, uint32_t iteration, uint32_t memory, uint32_t memMask) : algo(algorithm), iter(iteration), mem(memory), maskByte(memMask)
	{
	}

	operator xmrstak_algo_id() const
	{
		return algo;
	}

	xmrstak_algo_id id() const
	{
		return algo;
	}

	size_t Mem() const
	{
		if(algo == invalid_algo)
			return 0;
		else
			return mem;
	}

	uint32_t Iter() const
	{
		return iter;
	}

	uint32_t Mask() const
	{
		const uint32_t m_byte = cn_mask_bytes(algo);
		if(maskByte == 0)
			return ((mem - 1u) / m_byte) * m_byte;
		else
			return maskByte;
	}

	xmrstak_algo_id algo = invalid_algo;
	uint32_t iter = 0;
	size_t mem = 2u * 1024u * 1024u;
	uint32_t maskByte = 0;
};

// define aeon settings
constexpr size_t CRYPTONIGHT_LITE_MEMORY = 1 * 1024 * 1024;
constexpr uint32_t CRYPTONIGHT_LITE_MASK = 0xFFFF0;
constexpr uint32_t CRYPTONIGHT_LITE_ITER = 0x40000;

constexpr size_t CN_MEMORY = 2 * 1024 * 1024;
constexpr uint32_t CN_ITER = 0x80000;

constexpr uint32_t CN_GPU_MASK = 0x1FFFC0;
constexpr uint32_t CN_GPU_ITER = 0xC000;


constexpr size_t CRYPTONIGHT_HEAVY_MEMORY = 4 * 1024 * 1024;
constexpr uint32_t CRYPTONIGHT_HEAVY_MASK = 0x3FFFF0;
constexpr uint32_t CRYPTONIGHT_HEAVY_ITER = 0x40000;

constexpr uint32_t CRYPTONIGHT_GPU_MASK = 0x1FFFC0;
constexpr uint32_t CRYPTONIGHT_GPU_ITER = 0xC000;

constexpr uint32_t CRYPTONIGHT_MASARI_ITER = 0x40000;

constexpr uint32_t CRYPTONIGHT_SUPERFAST_ITER = 0x20000;

constexpr size_t CRYPTONIGHT_TURTLE_MEMORY = 256 * 1024;
constexpr uint32_t CRYPTONIGHT_TURTLE_MASK = 0x1FFF0;
constexpr uint32_t CRYPTONIGHT_TURTLE_ITER = 0x10000;
