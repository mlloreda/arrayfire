/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
//#include <kernel_headers/confidence_connected.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <cache.hpp>
#include <common/dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>

namespace opencl
{
namespace kernel
{
template <typename T>
void confidenceConnected(Param out,
                         const Param in, const Param seed,
                         const af::ccType p,
                         unsigned radius, unsigned multiplier, int iter)
{
    OPENCL_NOT_SUPPORTED("Cuda backend has no implementation yet");
}
}
}
