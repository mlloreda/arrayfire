/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/signal.h>
#include <af/defines.h>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <approx.hpp>

using af::dim4;
using namespace detail;

template<typename Ty, typename Tp>
static inline af_array approx1(const af_array in, const af_array pos,
                               const af_interp_type method, const float offGrid)
{
    return getHandle(approx1<Ty>(getArray<Ty>(in), getArray<Tp>(pos), method, offGrid));
}

template<typename Ty, typename Tp>
static inline af_array approx2(const af_array in, const af_array pos0, const af_array pos1,
                               const af_interp_type method, const float offGrid)
{
    return getHandle(approx2<Ty>(getArray<Ty>(in), getArray<Tp>(pos0), getArray<Tp>(pos1),
                                 method, offGrid));
}

af_err af_approx1(af_array *out, const af_array in, const af_array pos,
                  const af_interp_type method, const float offGrid)
{
    try {
        ARG_SETUP(in);
        ARG_SETUP(pos);

        const dim4 idims = in_info.dims();
        const dim4 pdims = pos_info.dims();

        // \NOTES(miguel)
        //
        // . What to do with these ARG_ASSERTS? Can we use one of the
        // ASSERT_TYPE macros? We can use AF_ASSERT() here.
        //
        ARG_ASSERT(1, in_info.isFloating());                         // Only floating and complex types
        ARG_ASSERT(2, pos_info.isRealFloating());                    // Only floating types
        ARG_ASSERT(1, in_info.isSingle() == pos_info.isSingle());    // Must have same precision
        ARG_ASSERT(1, in_info.isDouble() == pos_info.isDouble());    // Must have same precision
        // POS should either be (x, 1, 1, 1) or (1, idims[1], idims[2], idims[3])


        // \NOTES(miguel)
        //
        // . What to do with this DIM_ASSERT? new macro that accepts
        // condition along with fmt string? fmt_str = "`pos` must
        // either be a column, or its last three dims must match with
        // `in`"
        //
        // ASSERT_DIM_FMT(fmt_str, (pos_info.isColumn() || (pdims[1] == idims[1] && pdims[2] == idims[2] && pdims[3] == idims[3])));
        //
        DIM_ASSERT(2, pos_info.isColumn() ||
                      (pdims[1] == idims[1] && pdims[2] == idims[2] && pdims[3] == idims[3]));


        ARG_ASSERT(3, (method == AF_INTERP_LINEAR  ||
                       method == AF_INTERP_NEAREST ||
                       method == AF_INTERP_CUBIC   ||
                       method == AF_INTERP_CUBIC_SPLINE ||
                       method == AF_INTERP_LINEAR_COSINE ||
                       method == AF_INTERP_LOWER));

        if (idims.ndims() == 0 || pdims.ndims() ==  0) {
            return af_create_handle(out, 0, nullptr, in_info.getType());
        }

        af_array output;
        switch(in_info.getType()) {
            case f32: output = approx1<float  , float >(in, pos, method, offGrid);  break;
            case f64: output = approx1<double , double>(in, pos, method, offGrid);  break;
            case c32: output = approx1<cfloat , float >(in, pos, method, offGrid);  break;
            case c64: output = approx1<cdouble, double>(in, pos, method, offGrid);  break;
            default:  TYPE_ERROR(in);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_approx2(af_array *out, const af_array in, const af_array pos0, const af_array pos1,
                  const af_interp_type method, const float offGrid)
{
    try {
        ARG_SETUP(in);
        ARG_SETUP(pos0);
        ARG_SETUP(pos1);

        const dim4 idims = in_info.dims();
        const dim4 pdims = pos0_info.dims();
        const dim4 qdims = pos1_info.dims();

        ARG_ASSERT(1, in_info.isFloating()); // Only floating and complex types // \TODO(miguel)
        ARG_ASSERT(2, pos0_info.isRealFloating()); // Only floating types
        ARG_ASSERT(3, pos1_info.isRealFloating()); // Only floating types
        ARG_ASSERT(1, pos0_info.getType() == pos1_info.getType()); // Must have same type
        ARG_ASSERT(1, in_info.isSingle() == pos0_info.isSingle()); // Must have same precision
        ARG_ASSERT(1, in_info.isDouble() == pos0_info.isDouble()); // Must have same precision
        ASSERT_DIM(pos0, pos1);                           // POS0 and POS1 must have same dims

        // POS should either be (x, y, 1, 1) or (x, y, idims[2], idims[3])
        // \TODO(miguel) ASSERT_DIM_FMT(
        DIM_ASSERT(2, (pdims[2] == 1        && pdims[3] == 1) ||
                      (pdims[2] == idims[2] && pdims[3] == idims[3]));

        if (idims.ndims() == 0 || pdims.ndims() ==  0 || qdims.ndims() == 0) {
            return af_create_handle(out, 0, nullptr, in_info.getType());
        }

        af_array output;
        switch(in_info.getType()) {
            case f32: output = approx2<float  , float >(in, pos0, pos1, method, offGrid);  break;
            case f64: output = approx2<double , double>(in, pos0, pos1, method, offGrid);  break;
            case c32: output = approx2<cfloat , float >(in, pos0, pos1, method, offGrid);  break;
            case c64: output = approx2<cdouble, double>(in, pos0, pos1, method, offGrid);  break;
            default:  TYPE_ERROR(in);
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
