/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <math.hpp>                     // CUDA specific math functions

#include <Param.hpp>                    // This header has the declaration of structures
                                        // that are passed onto kernel. Operator overloads
                                        // for creating Param objects from cuda::Array<T>
                                        // objects is automatic, no special work is needed.
                                        // Hence, the CUDA kernel wrapper function takes in
                                        // Param and CParam(constant version of Param) instead
                                        // of cuda::Array<T>

#include <common/dispatch.hpp>                 // common utility header for CUDA & OpenCL backends
                                        // has the divup macro

#include <err_cuda.hpp>                 // CUDA specific error check functions and macros

#include <debug_cuda.hpp>               // For Debug only related CUDA validations

namespace cuda
{

namespace kernel
{

static const unsigned TX = 16;          // Kernel Launch Config Values
static const unsigned TY = 16;          // Kernel Launch Config Values
const int threadsPerBlock = TX*TY;


template <typename T>                   // CUDA kernel wrapper function
void confidence_connected(cuda::Param<T> out, cuda::CParam<T> in, const af_cc_type p, cuda::CParam<T> seed, unsigned radius, unsigned multiplier, int iter)
{


}

}

}
