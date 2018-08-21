/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <af/image.h>
#include <handle.hpp>
#include <common/err_common.hpp>
#include <scan.hpp>

using af::dim4;
using namespace detail;

template<typename To, typename Ti>
static af_array sat(const af_array& in)
{
    const Array<To> input = castArray<To>(in);

    Array<To> hprefix_scan = scan<af_add_t, To, To>(input, 0);
    Array<To> vprefix_scan = scan<af_add_t, To, To>(hprefix_scan, 1);

    return getHandle<To>(vprefix_scan);
}

af_err af_sat(af_array* out, const af_array in)
{
    try {
        ARG_SETUP(in);
        ASSERT_NDIM_GT(in, 1);

        af_array output = 0;
        switch(in_info.getType()) {
            case f64: output = sat<double, double>(in); break;
            case f32: output = sat<float , float >(in); break;
            case s32: output = sat<int   , int   >(in); break;
            case u32: output = sat<uint  , uint  >(in); break;
            case  b8: output = sat<int   , char  >(in); break;
            case  u8: output = sat<uint  , uchar >(in); break;
            case s64: output = sat<intl  , intl  >(in); break;
            case u64: output = sat<uintl , uintl >(in); break;
            case s16: output = sat<int   , short >(in); break;
            case u16: output = sat<uint  , ushort>(in); break;
            default: TYPE_ERROR(in);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
