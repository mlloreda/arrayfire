/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace cuda
{
template<typename T>
Array<T> confidence_connected(const Array<T> &in, const af_cc_type method, Array<T> seed, unsigned radius, unsigned multiplier, int iter);
}
