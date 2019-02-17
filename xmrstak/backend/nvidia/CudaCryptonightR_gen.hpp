#pragma once

#include "xmrstak/backend/cryptonight.hpp"

#include <stdint.h>
#include <vector>
#include <string>


namespace xmrstak
{
namespace nvidia
{

void CryptonightR_get_program(std::vector<char>& ptx, std::string& lowered_name,
	const xmrstak_algo algo, uint64_t height, int arch_major, int arch_minor, bool background = false);


} // namespace xmrstak
} //namespace nvidia

