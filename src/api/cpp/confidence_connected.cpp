/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>       // af::array class is declared here

#include <af/image.h>        // Include the header related to the function

#include "error.hpp"        // AF_THROW macro to use error code C-API
                            // is going to return and throw corresponding
                            // exceptions if call isn't a success

namespace af
{

array confidence(const array& in, const af_cc_type p, array seed, unsigned radius, unsigned multiplier, int iter)
{
    // create a temporary af_array handle
    af_array temp = 0;

    // call C-API function
    AF_THROW( af_confidence_connected(&temp, in.get(),
                                      p,
                                      seed.get(),
                                      radius,
                                      multiplier,
                                      iter) );

    // array::get() returns af_array handle for the corresponding cpp af::array
    return array(temp);
}

}
