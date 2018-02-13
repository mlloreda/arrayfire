/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>

namespace cuda
{
namespace kernel
{
template <typename T>
void confidenceConnected(Param<T> out,
                         CParam<T> in, CParam<T> seed,
                         const af::ccType p,
                         unsigned radius, unsigned multiplier, int iter)
{
    CUDA_NOT_SUPPORTED("Cuda backend has no implementation yet");
}
}
}
