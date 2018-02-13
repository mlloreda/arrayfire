/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/image.h>
#include "error.hpp"

namespace af
{
af::array confidenceCC(const af::array& in, const af::array& seed,
                       const af::ccType ccKind, const unsigned radius,
                       const unsigned multiplier, const int iter)
{
    af_array temp = 0;
    AF_THROW(af_confidence_cc(&temp, in.get(), seed.get(), ccKind, radius, multiplier, iter));
    return array(temp);
}
}
