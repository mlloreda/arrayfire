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
        ARG_SETUP(x_src); ARG_SETUP(y_src);
        ARG_SETUP(x_dst); ARG_SETUP(y_dst);
        ASSERT_TYPE(x_src, TYPES(f32));
        ASSERT_TYPE(y_src, TYPES(f32));
        ASSERT_TYPE(x_dst, TYPES(f32));
        ASSERT_TYPE(y_dst, TYPES(f32));

        ASSERT_DIM_EQ(x_src, 0, 0);
        ASSERT_DIM_EQ(y_src, 0, 0); // \TODO(miguel) append to first dim assert
        ASSERT_DIM_EQ(x_dst, 0, 0);
        ASSERT_DIM_EQ(y_dst, 0, 0);

        ASSERT_NDIM_GT(x_dst, 0);

        ARG_ASSERT(4, (y_dst_info.dims()[0] == x_dst_info.dims()[0])); // \TODO(miguel)
        ARG_ASSERT(5, (inlier_thr >= 0.1f)); // \TODO(miguel)?
        ARG_ASSERT(6, (iterations > 0));     // \TODO(miguel)?

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
