/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/lapack.h>
#include <af/defines.h>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <qr.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static inline void qr(af_array *q, af_array *r, af_array *tau, const af_array in)
{
    Array<T> qArray = createEmptyArray<T>(af::dim4());
    Array<T> rArray = createEmptyArray<T>(af::dim4());
    Array<T> tArray = createEmptyArray<T>(af::dim4());

    qr<T>(qArray, rArray, tArray, getArray<T>(in));

    *q   = getHandle(qArray);
    *r   = getHandle(rArray);
    *tau = getHandle(tArray);
}

template<typename T>
static inline af_array qr_inplace(af_array in)
{
    return getHandle(qr_inplace<T>(getWritableArray<T>(in)));
}

af_err af_qr(af_array *q, af_array *r, af_array *tau, const af_array in)
{
    try {
        ARG_SETUP(in);

        if (in_info.ndims() > 2) {
            AF_ERROR("qr can not be used in batch mode", AF_ERR_BATCH);
        }

        const af_dtype in_type = in_info.getType();
        if (in_info.ndims() == 0) {
            AF_CHECK(af_create_handle(q,   0, nullptr, in_type));
            AF_CHECK(af_create_handle(r,   0, nullptr, in_type));
            AF_CHECK(af_create_handle(tau, 0, nullptr, in_type));
            return AF_SUCCESS;
        }

        ARG_ASSERT(3, in_info.isFloating());                       // Only floating and complex types

        switch(in_type) {
            case f32: qr<float  >(q, r, tau, in);  break;
            case f64: qr<double >(q, r, tau, in);  break;
            case c32: qr<cfloat >(q, r, tau, in);  break;
            case c64: qr<cdouble>(q, r, tau, in);  break;
            default:  TYPE_ERROR(in);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_qr_inplace(af_array *tau, af_array in)
{
    try {
        ARG_SETUP(in);
        ARG_ASSERT(1, in_info.isFloating()); // Only floating and complex types
        if (in_info.ndims() > 2) {
            AF_ERROR("qr can not be used in batch mode", AF_ERR_BATCH);
        }

        if (in_info.ndims() == 0) {
            return af_create_handle(tau, 0, nullptr, in_info.getType());
        }

        af_array out;
        switch(in_info.getType()) {
            case f32: out = qr_inplace<float  >(in);  break;
            case f64: out = qr_inplace<double >(in);  break;
            case c32: out = qr_inplace<cfloat >(in);  break;
            case c64: out = qr_inplace<cdouble>(in);  break;
            default:  TYPE_ERROR(in);
        }
        if (tau != NULL) {
            std::swap(*tau, out);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}
