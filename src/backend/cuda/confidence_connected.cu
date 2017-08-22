/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>                    // header with cuda backend specific
                                        // Array class implementation that inherits
                                        // ArrayInfo base class

#include <confidence_connected.hpp>     // cuda backend function header

#include <err_cuda.hpp>                 // error check functions and Macros
                                        // specific to cuda backend

#include <kernel/confidence_connected.hpp>   // this header under the folder src/cuda/kernel
                                        // defines the CUDA kernel and its wrapper
                                        // function to which the main computation of your
                                        // algorithm should be relayed to

using af::dim4;

namespace cuda
{

template<typename T>
Array<T> confidence_connected(const Array<T> &in,
                              const af_cc_type method,
                              Array<T> seed,
                              unsigned radius,
                              unsigned multiplier,
                              int iter)
{
    dim4 outputDims;                    // this should be '= in.dims();' in most cases
                                        // but would definitely depend on the type of
                                        // algorithm you are implementing.

    // std::cout << "dims(0): " << in.dims() << std::endl;
    Array<T> out = createEmptyArray<T>(in.dims());
                                        // Please use the create***Array<T> helper
                                        // functions defined in Array.hpp to create
                                        // different types of Arrays. Please check the
                                        // file to know what are the different types you
                                        // can create.

    // Relay the actual computation to CUDA kernel wrapper
    cuda::kernel::confidence_connected<T>(out, in, method, seed, radius, multiplier, iter);

    return out;                         // return the result
}


#define INSTANTIATE(T)  \
    template Array<T> confidence_connected<T>(const Array<T> &in, const af_cc_type method, const Array<T> seed, unsigned radius, unsigned multiplier, int iter);

// INSTANTIATIONS for all the types which
// are present in the switch case statement
// in src/api/c/exampleFunction.cpp should be available
// INSTANTIATE(float)
// INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
// INSTANTIATE(short)
// INSTANTIATE(ushort)
INSTANTIATE(uchar)

}
