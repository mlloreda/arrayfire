/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/signal.h>
#include <af/arith.h>
#include <handle.hpp>
#include <common/err_common.hpp>
#include <backend.hpp>
#include <convolve.hpp>
#include <iir.hpp>

#include <cstdio>

using std::vector;
using af::dim4;
using namespace detail;

af_err af_fir(af_array *y, const af_array b, const af_array x)
{
    try {
        af_array out;
        AF_CHECK(af_convolve1(&out, x, b, AF_CONV_EXPAND, AF_CONV_AUTO));

        ARG_SETUP(x);

        dim4 xdims = x_info.dims();
        af_seq seqs[] = {af_span, af_span, af_span, af_span};
        seqs[0].begin = 0;
        seqs[0].end = xdims[0] - 1;
        seqs[0].step = 1;
        af_array res;
        AF_CHECK(af_index(&res, out, 4, seqs));
        AF_CHECK(af_release_array(out));
        std::swap(*y, res);

    } CATCHALL;
    return AF_SUCCESS;
}

template<typename T>
inline static af_array iir(const af_array b, const af_array a, const af_array x)
{
    return getHandle(iir<T>(getArray<T>(b),
                            getArray<T>(a),
                            getArray<T>(x)));
}

af_err af_iir(af_array *y, const af_array b, const af_array a, const af_array x)
{
    try {
        ARG_SETUP(a);
        ARG_SETUP(b);
        ARG_SETUP(x);

        ASSERT_TYPE(x, TYPES(f32, f64, c32, c64));
        ASSERT_TYPE_EQ(x, a);
        ASSERT_TYPE_EQ(x, b);

        ARG_ASSERT(1, b_info.ndims() == a_info.ndims());

        if (x_info.ndims() == 0) {
            return af_retain_array(y, x);
        }

        if (x_info.ndims() > 1) {
            if (b_info.ndims() > 1) {
                for (int i = 1; i < 3; i++) {
                    ARG_ASSERT(1, b_info.dims()[i] == x_info.dims()[i]);
                }
            }
        }

        // If only a0 is available, just normalize b and perform fir
        if (a_info.dims()[0] == 1) {
            af_array bnorm = 0;
            AF_CHECK(af_div(&bnorm, b, a, true));
            AF_CHECK(af_fir(y, bnorm, x));
            AF_CHECK(af_release_array(bnorm));
            return AF_SUCCESS;
        }

        af_array res;
        switch (x_info.getType()) {
        case f32: res = iir<float  >(b, a, x); break;
        case f64: res = iir<double >(b, a, x); break;
        case c32: res = iir<cfloat >(b, a, x); break;
        case c64: res = iir<cdouble>(b, a, x); break;
        default: TYPE_ERROR(x);
        }

        std::swap(*y, res);
    } CATCHALL;
    return AF_SUCCESS;
}
