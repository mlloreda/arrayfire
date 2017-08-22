/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>                    // header with cpu backend specific
                                        // Array class implementation that inherits
                                        // ArrayInfo base class


#include <confidence_connected.hpp>          // cpu backend function header
#include <kernel/confidence_connected.hpp>   // Function implementation header

#include <err_cpu.hpp>                  // error check functions and Macros
                                        // specific to cpu backend

// From regions()...
// #include <af/dim4.hpp>
// #include <Array.hpp>
// #include <regions.hpp>
// #include <err_cpu.hpp>
// #include <math.hpp>
// #include <map>
// #include <set>
// #include <algorithm>
// #include <platform.hpp>
// #include <queue.hpp>
// #include <kernel/regions.hpp>

#include <map>

using af::dim4;

namespace cpu
{

template<typename T>
Array<T> confidence_connected(const Array<T> &in, const af_cc_type method, Array<T> seed, unsigned radius, unsigned multiplier, int iter)
{
    in.eval();                          // All input Arrays should call eval mandatorily
                                        // in CPU backend function implementations. Since
                                        // the cpu fns are asynchronous launches, any Arrays
                                        // that are either views/JIT nodes needs to evaluated
                                        // before they are passed onto functions that are
                                        // enqueued onto the queues.

    dim4 outputDims = in.dims();        // this should be '= in.dims();' in most cases
                                        // but would definitely depend on the type of
                                        // algorithm you are implementing.

    Array<T> out = createEmptyArray<T>(outputDims);
                                        // Please use the create***Array<T> helper
                                        // functions defined in Array.hpp to create
                                        // different types of Arrays. Please check the
                                        // file to know what are the different types you
                                        // can create.

    // Enqueue the function call on the worker thread
    // This code will be present in src/backend/cpu/kernel/exampleFunction.hpp

    getQueue().enqueue(kernel::confidence_connected<T>, out, in, method, seed, radius, multiplier, iter);

    return out;                         // return the result
}


#define INSTANTIATE(T)  \
    template Array<T> confidence_connected<T>(const Array<T> &in, const af_cc_type method, Array<T> seed, unsigned radius, unsigned multiplier, int iter);

// INSTANTIATIONS for all the types which
// are present in the switch case statement
// in src/api/c/exampleFunction.cpp should be available
INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(char)

}
