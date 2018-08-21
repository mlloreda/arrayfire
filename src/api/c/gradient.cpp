/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/image.h>
#include <af/defines.h>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <gradient.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static inline void gradient(af_array *grad0, af_array *grad1, const af_array in)
{
    gradient<T>(getWritableArray<T>(*grad0), getWritableArray<T>(*grad1), getArray<T>(in));
}

af_err af_gradient(af_array *grows, af_array *gcols, const af_array in)
{
    try {
        ARG_SETUP(in);
        DIM_ASSERT(2, in_info.elements() > 0); // \TODO(miguel);

        af_array grad0;
        af_array grad1;
        const dim4 in_dims = in_info.dims();
        const af_dtype in_type = in_info.getType();
        AF_CHECK(af_create_handle(&grad0, in_dims.ndims(), in_dims.get(), in_type));
        AF_CHECK(af_create_handle(&grad1, in_dims.ndims(), in_dims.get(), in_type));

        switch(in_type) {
            case f32: gradient<float  >(&grad0, &grad1, in);  break;
            case c32: gradient<cfloat >(&grad0, &grad1, in);  break;
            case f64: gradient<double >(&grad0, &grad1, in);  break;
            case c64: gradient<cdouble>(&grad0, &grad1, in);  break;
            default:  TYPE_ERROR(in);
        }
        std::swap(*grows, grad0);
        std::swap(*gcols, grad1);
    }
    CATCHALL;

    return AF_SUCCESS;
}
