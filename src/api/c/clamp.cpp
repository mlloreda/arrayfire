/*******************************************************
 * Copyright (c) 2014, ArrayFire
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

#include <arith.hpp>
#include <logic.hpp>

using namespace detail;
using af::dim4;

template<typename T>
static inline af_array clampOp(const af_array in,
                               const af_array lo,
                               const af_array hi,
                               const dim4 &odims)
{
    const Array<T> L = castArray<T>(lo);
    const Array<T> H = castArray<T>(hi);
    const Array<T> I = castArray<T>(in);
    return getHandle(arithOp<T, af_min_t>(arithOp<T, af_max_t>(I, L, odims), H, odims));
}

af_err af_clamp(af_array *out, const af_array in,
                const af_array lo, const af_array hi, const bool batch)
{
    try {
        ARG_SETUP(in);
        ARG_SETUP(lo);
        ARG_SETUP(hi);
        ASSERT_DIM(lo, hi);
        ASSERT_TYPE_EQ(lo, hi);

        const     dim4 odims = getOutDims(in_info.dims(), lo_info.dims(), batch);
        const af_dtype otype = implicit(in_info.getType(), lo_info.getType());
        af_array res;
        switch (otype) {
        case f32: res = clampOp<float  >(in, lo, hi, odims); break;
        case f64: res = clampOp<double >(in, lo, hi, odims); break;
        case c32: res = clampOp<cfloat >(in, lo, hi, odims); break;
        case c64: res = clampOp<cdouble>(in, lo, hi, odims); break;
        case s32: res = clampOp<int    >(in, lo, hi, odims); break;
        case u32: res = clampOp<uint   >(in, lo, hi, odims); break;
        case u8 : res = clampOp<uchar  >(in, lo, hi, odims); break;
        case b8 : res = clampOp<char   >(in, lo, hi, odims); break;
        case s64: res = clampOp<intl   >(in, lo, hi, odims); break;
        case u64: res = clampOp<uintl  >(in, lo, hi, odims); break;
        case s16: res = clampOp<short  >(in, lo, hi, odims); break;
        case u16: res = clampOp<ushort >(in, lo, hi, odims); break;
        default: UNSUPPORTED_TYPE(otype);
        }

        std::swap(*out, res);
    }
    CATCHALL;
    return AF_SUCCESS;
}
