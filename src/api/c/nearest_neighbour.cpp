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
#include <af/vision.h>
#include <handle.hpp>
#include <common/err_common.hpp>
#include <backend.hpp>
#include <nearest_neighbour.hpp>

using af::dim4;
using namespace detail;

template<typename Ti, typename To>
static void nearest_neighbour(af_array* idx, af_array* dist,
        const af_array query, const af_array train,
        const dim_t dist_dim, const uint n_dist,
        const af_match_type dist_type)
{
    Array<uint> oIdxArray = createEmptyArray<uint>(af::dim4());
    Array<To>  oDistArray = createEmptyArray<To>(af::dim4());

    nearest_neighbour<Ti, To>(oIdxArray, oDistArray, getArray<Ti>(query), getArray<Ti>(train),
                              dist_dim, n_dist, dist_type);

    *idx  = getHandle<uint>(oIdxArray);
    *dist = getHandle<To>(oDistArray);
}

af_err af_nearest_neighbour(af_array* idx, af_array* dist,
        const af_array query, const af_array train,
        const dim_t dist_dim, const uint n_dist,
        const af_match_type dist_type)
{
    try {
        ARG_SETUP(query);
        ARG_SETUP(train);
        ASSERT_TYPE_EQ(query, train);
        ASSERT_DIM_EQ(query, 2, 1);
        ASSERT_DIM_EQ(query, 3, 1);
        ASSERT_DIM_EQ(train, 2, 1);
        ASSERT_DIM_EQ(train, 3, 1);

        const dim4 query_dims = query_info.dims();
        const dim4 train_dims = train_info.dims();
        const uint train_samples = (dist_dim == 0) ? 1 : 0;
        DIM_ASSERT(2, query_dims[dist_dim] == train_dims[dist_dim]);
        DIM_ASSERT(4, (dist_dim == 0 || dist_dim == 1));
        DIM_ASSERT(5, n_dist > 0 && n_dist <= (uint)train_dims[train_samples]);
        ARG_ASSERT(6, dist_type == AF_SAD || dist_type == AF_SSD || dist_type == AF_SHD);

        // For Hamming, only u8, u16, u32 and u64 allowed.
        af_array oIdx;
        af_array oDist;

        if (dist_type == AF_SHD) {
            ASSERT_TYPE(query, TYPES(u8, u16, u32, u64));
            switch(query_info.getType()) {
                case u8:  nearest_neighbour<uchar , uint>(&oIdx, &oDist, query, train, dist_dim, n_dist, AF_SHD); break;
                case u16: nearest_neighbour<ushort, uint>(&oIdx, &oDist, query, train, dist_dim, n_dist, AF_SHD); break;
                case u32: nearest_neighbour<uint  , uint>(&oIdx, &oDist, query, train, dist_dim, n_dist, AF_SHD); break;
                case u64: nearest_neighbour<uintl , uint>(&oIdx, &oDist, query, train, dist_dim, n_dist, AF_SHD); break;
                default : TYPE_ERROR(query);
            }
        } else {
            switch(query_info.getType()) {
                case f32: nearest_neighbour<float , float >(&oIdx, &oDist, query, train, dist_dim, n_dist, dist_type); break;
                case f64: nearest_neighbour<double, double>(&oIdx, &oDist, query, train, dist_dim, n_dist, dist_type); break;
                case s32: nearest_neighbour<int   , int   >(&oIdx, &oDist, query, train, dist_dim, n_dist, dist_type); break;
                case u32: nearest_neighbour<uint  , uint  >(&oIdx, &oDist, query, train, dist_dim, n_dist, dist_type); break;
                case s64: nearest_neighbour<intl  , intl  >(&oIdx, &oDist, query, train, dist_dim, n_dist, dist_type); break;
                case u64: nearest_neighbour<uintl , uintl >(&oIdx, &oDist, query, train, dist_dim, n_dist, dist_type); break;
                case s16: nearest_neighbour<short , int   >(&oIdx, &oDist, query, train, dist_dim, n_dist, dist_type); break;
                case u16: nearest_neighbour<ushort, uint  >(&oIdx, &oDist, query, train, dist_dim, n_dist, dist_type); break;
                case u8:  nearest_neighbour<uchar , uint  >(&oIdx, &oDist, query, train, dist_dim, n_dist, dist_type)
; break;
                default : TYPE_ERROR(query);
            }
        }
        std::swap(*idx, oIdx);
        std::swap(*dist, oDist);
    }
    CATCHALL;

    return AF_SUCCESS;
}
