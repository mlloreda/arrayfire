/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace cpu
{
template<typename T>
Array<T> confidence_connected(const Array<T> &in, const Array<T>& seed,
                              const af::ccType method,
                              unsigned radius, unsigned multiplier, int iter);
}
