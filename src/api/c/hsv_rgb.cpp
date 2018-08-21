/*******************************************************
* Copyright (c) 2014, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/image.h>
#include <handle.hpp>
#include <common/err_common.hpp>
#include <backend.hpp>
#include <hsv_rgb.hpp>

using af::dim4;
using namespace detail;

template<typename T, bool isHSV2RGB>
static af_array convert(const af_array& in)
{
    const Array<T> input = getArray<T>(in);
    if (isHSV2RGB) {
        return getHandle<T>(hsv2rgb<T>(input));
    }
    else {
        return getHandle<T>(rgb2hsv<T>(input));
    }
}

template<bool isHSV2RGB>
af_err convert(af_array* out, const af_array& in)
{
    try {
        ARG_SETUP(in);
        ASSERT_NDIM_GT(in, 2);

        if (in_info.ndims() == 0) {
            return af_create_handle(out, 0, nullptr, in_info.getType());
        }

        af_array output = 0;
        switch (in_info.getType()) {
            case f64: output = convert<double, isHSV2RGB>(in); break;
            case f32: output = convert<float , isHSV2RGB>(in); break;
            default: TYPE_ERROR(in); break;
        }
        std::swap(*out, output);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_hsv2rgb(af_array* out, const af_array in)
{
    return convert<true>(out, in);
}

af_err af_rgb2hsv(af_array* out, const af_array in)
{
    return convert<false>(out, in);
}
