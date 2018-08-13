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
#include <af/vision.h>
#include <af/random.h>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <homography.hpp>

using af::dim4;
using std::vector;
using namespace detail;

template<typename T>
static inline void homography(af_array &H, int &inliers,
                              const af_array x_src, const af_array y_src,
                              const af_array x_dst, const af_array y_dst,
                              const af_homography_type htype, const float inlier_thr,
                              const unsigned iterations)
{
    Array<T> bestH = createEmptyArray<T>(af::dim4(3, 3));
    af_array initial;
    unsigned d = (iterations + 256 - 1) / 256;
    dim_t rdims[] = {4, d * 256};
    AF_CHECK(af_randu(&initial, 2, rdims, f32));
    inliers = homography<T>(bestH,
                            getArray<float>(x_src), getArray<float>(y_src),
                            getArray<float>(x_dst), getArray<float>(y_dst),
                            getArray<float>(initial),
                            htype, inlier_thr, iterations);
    AF_CHECK(af_release_array(initial));

    H = getHandle<T>(bestH);
}

af_err af_homography(af_array *H, int *inliers,
                     const af_array x_src, const af_array y_src,
                     const af_array x_dst, const af_array y_dst,
                     const af_homography_type htype, const float inlier_thr,
                     const unsigned iterations, const af_dtype otype)
{
    try {
        ARG_SETUP(x_src);
        ARG_SETUP(y_src);
        ARG_SETUP(x_dst);
        ARG_SETUP(y_dst);

        ASSERT_TYPE(x_src, TYPES(f32));
        ASSERT_TYPE(y_src, TYPES(f32));
        ASSERT_TYPE(x_dst, TYPES(f32));
        ASSERT_TYPE(y_dst, TYPES(f32));

        dim4 xsdims  = x_src_info.dims();
        dim4 ysdims  = y_src_info.dims();
        dim4 xddims  = x_dst_info.dims();
        dim4 yddims  = y_dst_info.dims();

        // \TODO enable DIM assertions!
        // int dim_idx = 0;
        // ASSERT_DIM_GT(dim_idx, 0, x_src);
        // ASSERT_DIM_GT(dim_idx, 0, x_dst);
        // ASSERT_DIM_GT(dim_idx, 0, x_src, x_dst); // TODO multiple arrays
        // ASSERT_DIM_GT(x_src, dim_idx, 0);
        // ASSERT_DIM(x_src[0], y_src[0]) ?????
        // ASSERT_DIM_GT(0, 0, x_src, x_dst);

        ARG_ASSERT(1, (xsdims[0] > 0));
        ARG_ASSERT(2, (ysdims[0] == xsdims[0]));
        ARG_ASSERT(3, (xddims[0] > 0));
        ARG_ASSERT(4, (yddims[0] == xddims[0]));
        ARG_ASSERT(5, (inlier_thr >= 0.1f));
        ARG_ASSERT(6, (iterations > 0));

        af_array outH;
        int outInl;
        switch(otype) {
            case f32: homography<float >(outH, outInl, x_src, y_src, x_dst, y_dst, htype, inlier_thr, iterations);  break;
            case f64: homography<double>(outH, outInl, x_src, y_src, x_dst, y_dst, htype, inlier_thr, iterations);  break;
            default:  UNSUPPORTED_TYPE(otype);
        }
        std::swap(*H, outH);
        std::swap(*inliers, outInl);
    }
    CATCHALL;

    return AF_SUCCESS;
}
