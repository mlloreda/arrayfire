/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <af/array.h>
#include <af/defines.h>
#include <af/arith.h>
#include <af/data.h>
#include <common/ArrayInfo.hpp>
#include <optypes.hpp>
#include <implicit.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>

#include <select.hpp>

using namespace detail;
using af::dim4;

template<typename T>
void replace(af_array a, const af_array cond, const af_array b)
{
    select(getCopyOnWriteArray<T>(a), getArray<char>(cond), getArray<T>(a), getArray<T>(b));
}

af_err af_replace(af_array a, const af_array cond, const af_array b)
{
    try {
        ARG_SETUP(a);
        ARG_SETUP(b);
        ARG_SETUP(cond);

        if (cond_info.ndims() == 0) {
            return AF_SUCCESS;
        }

        ASSERT_TYPE_EQ(a, b);
        ASSERT_TYPE(cond, TYPES(b8));

        DIM_ASSERT(1, a_info.ndims() >= b_info.ndims());
        DIM_ASSERT(1, cond_info.ndims() == std::min(a_info.ndims(), b_info.ndims()));

        dim4 adims = a_info.dims();
        dim4 bdims = b_info.dims();
        dim4 cdims = cond_info.dims();

        for (int i = 0; i < 4; i++) {
            DIM_ASSERT(1, cdims[i] == std::min(adims[i], bdims[i]));
            DIM_ASSERT(2, adims[i] == bdims[i] || bdims[i] == 1);
        }

        switch (a_info.getType()) {
        case f32: replace<float  >(a, cond, b); break;
        case f64: replace<double >(a, cond, b); break;
        case c32: replace<cfloat >(a, cond, b); break;
        case c64: replace<cdouble>(a, cond, b); break;
        case s32: replace<int    >(a, cond, b); break;
        case u32: replace<uint   >(a, cond, b); break;
        case s64: replace<intl   >(a, cond, b); break;
        case u64: replace<uintl  >(a, cond, b); break;
        case s16: replace<short  >(a, cond, b); break;
        case u16: replace<ushort >(a, cond, b); break;
        case u8:  replace<uchar  >(a, cond, b); break;
        case b8:  replace<char   >(a, cond, b); break;
        default:  TYPE_ERROR(a);
        }

    } CATCHALL;
    return AF_SUCCESS;
}

template<typename T>
void replace_scalar(af_array a, const af_array cond, const double b)
{
    select_scalar<T, false>(getCopyOnWriteArray<T>(a), getArray<char>(cond), getArray<T>(a), b);
}

af_err af_replace_scalar(af_array a, const af_array cond, const double b)
{
    try {
        ARG_SETUP(a);
        ARG_SETUP(cond);

        ASSERT_TYPE(cond, TYPES(b8));

        DIM_ASSERT(1, cond_info.ndims() == a_info.ndims());

        dim4 adims = a_info.dims();
        dim4 cdims = cond_info.dims();

        for (int i = 0; i < 4; i++) {
            DIM_ASSERT(1, cdims[i] == adims[i]);
        }

        switch (a_info.getType()) {
        case f32: replace_scalar<float  >(a, cond, b); break;
        case f64: replace_scalar<double >(a, cond, b); break;
        case c32: replace_scalar<cfloat >(a, cond, b); break;
        case c64: replace_scalar<cdouble>(a, cond, b); break;
        case s32: replace_scalar<int    >(a, cond, b); break;
        case u32: replace_scalar<uint   >(a, cond, b); break;
        case s64: replace_scalar<intl   >(a, cond, b); break;
        case u64: replace_scalar<uintl  >(a, cond, b); break;
        case s16: replace_scalar<short  >(a, cond, b); break;
        case u16: replace_scalar<ushort >(a, cond, b); break;
        case u8:  replace_scalar<uchar  >(a, cond, b); break;
        case b8:  replace_scalar<char   >(a, cond, b); break;
        default:  TYPE_ERROR(a);
        }

    } CATCHALL;
    return AF_SUCCESS;
}
