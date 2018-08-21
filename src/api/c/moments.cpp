/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/image.h>
#include <af/index.h>
#include <af/data.h>

#include <common/ArrayInfo.hpp>
#include <common/graphics_common.hpp>
#include <common/err_common.hpp>
#include <backend.hpp>
#include <moments.hpp>
#include <handle.hpp>
#include <reorder.hpp>
#include <tile.hpp>
#include <join.hpp>
#include <cast.hpp>
#include <arith.hpp>

#include <limits>
#include <vector>

using af::dim4;

using std::vector;
using namespace detail;

template<typename T>
static inline void moments(af_array *out, const af_array in, af_moment_type moment)
{
    Array<float> temp = moments<T>(getArray<T>(in), moment);
    *out = getHandle<float>(temp);
}

af_err af_moments(af_array *out, const af_array in, const af_moment_type moment)
{
    try {
        ARG_SETUP(in);

        switch(in_info.getType()) {
            case f32: moments<float>          (out, in, moment); break;
            case f64: moments<double>         (out, in, moment); break;
            case u32: moments<unsigned>       (out, in, moment); break;
            case s32: moments<int>            (out, in, moment); break;
            case u16: moments<unsigned short> (out, in, moment); break;
            case s16: moments<short>          (out, in, moment); break;
            case b8:  moments<char>           (out, in, moment); break;
            default:  TYPE_ERROR(in);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<typename T>
static inline void moment_copy(double* out, const af_array moments)
{
    ARG_SETUP(moments);
    vector<T> h_moments(moments_info.elements());
    copyData(h_moments.data(), moments);

    // convert to double
    copy(begin(h_moments), end(h_moments), out);
}

af_err af_moments_all(double* out, const af_array in, const af_moment_type moment)
{
    try {
        ARG_SETUP(in);
        ASSERT_DIM_EQ(in, 2, 1);
        ASSERT_DIM_EQ(in, 3, 1);

        af_array moments_arr;
        AF_CHECK(af_moments(&moments_arr, in, moment));
        moment_copy<float>(out, moments_arr);
        AF_CHECK(af_release_array(moments_arr));
    }
    CATCHALL;

    return AF_SUCCESS;
}
