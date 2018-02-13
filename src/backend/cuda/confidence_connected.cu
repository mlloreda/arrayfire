/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <confidence_connected.hpp>
#include <err_cuda.hpp>
#include <kernel/confidence_connected.hpp>

namespace cuda
{
template<typename T>
Array<T> confidenceConnected(const Array<T>& in, const Array<T>& seed,
                             const af::ccType method,
                             unsigned radius, unsigned multiplier, int iter)
{
    Array<T> out = createEmptyArray<T>(in.dims());

    kernel::confidenceConnected<T>(out, in, seed, method, radius, multiplier, iter);

    return out;
}


#define INSTANTIATE(T)  \
template Array<T> confidenceConnected<T>(const Array<T> &in, const Array<T>& seed, \
        const af::ccType method, unsigned radius, unsigned multiplier, int iter);

INSTANTIATE(uchar)
}
