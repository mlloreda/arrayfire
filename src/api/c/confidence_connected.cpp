/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/arith.h>
#include <af/data.h>
#include <af/image.h>
#include <af/statistics.h>

#include <Array.hpp>
#include <backend.hpp>
#include <common/err_common.hpp>
#include <confidence_connected.hpp>
#include <handle.hpp>
#include <index.hpp>

using namespace detail;

template<typename T>
af_array confidence(const af_array& in, const af_array& seed, const af_cc_type& cc_type, unsigned radius, unsigned multiplier, int iter)
{
    return getHandle<T>(confidenceConnected(getArray<T>(in), getArray<T>(seed), cc_type, radius, multiplier, iter));
}

template<typename T>
af_array ConfidenceConnectedComponents(const Array<T> in,
                                       const int seedRow, const int seedCol,
                                       const int radius, const int iterations)
{
    // Array<T> seedIntensity = detail::constant(in(seedRow, seedCol), 1, 1);
#if 0
    Array<T> seedNeighborhood = in(0, 10);
#endif
    return getHandle<T>(in);
}


template<typename T>
af_array ccHelper(const Array<T> in, Array<T> seed,
                  const af_cc_type cc_type,
                  unsigned radius, unsigned mult, unsigned iter)
{
    std::cout << "Hello from ccHelper()\n";

    // (af_span, af_span, af_span, af_span)
    // af_err err = af_create_indexers(&indexers);
    af_index_t indexers[4];
    for (int i = 0; i < 4; ++i) {
        indexers[i].idx.seq = af_span;
        indexers[i].isSeq   = true;
        indexers[i].isBatch = false;
    }

    // af_array flat;
    // af_flat(&flat, getHandle(seed));
    // err = af_set_array_indexer(indexers, seed, 0);
    // err = af_set_array_indexer(indexers, seed, 1);

    // dim_t dimtest[] = { 1 };
    // // af_array row;
    // af_create_array(&row, getHandle(seed), 0, dimtest, s32);

    // \TODO
    int row = 2;
    int col = 2;

    indexers[0].isSeq   = true;
    indexers[0].idx.seq = af_make_seq(row, col, 1);
    indexers[0].isBatch = false;
    indexers[1].isSeq   = true;
    indexers[1].idx.seq = af_make_seq(row, col, 1);
    indexers[1].isBatch = false;

    // indexers[0].idx.arr = getHandle(seed);
    // indexers[0].isSeq   = false;
    // indexers[0].isBatch = false;
    // indexers[1].idx.arr = getHandle(seed);
    // indexers[1].isSeq   = false;
    // indexers[1].isBatch = false;

    // Get seed intensity value
    Array<T> seedIntensity = index<T>(in, indexers);

    std::cout << "seedIntensity: \n";
    af_print_array(getHandle(seedIntensity));

    // Get seed neighborhood
    indexers[0].isSeq   = true;
    indexers[0].idx.seq = af_make_seq(row-1, col+1, 1);
    indexers[0].isBatch = false;
    indexers[1].isSeq   = true;
    indexers[1].idx.seq = af_make_seq(row-1, col+1, 1);
    indexers[1].isBatch = false;

    Array<T> seedNeighborhood = index<T>(in, indexers);
    std::cout << "seedNeighborhood: \n";
    af_print_array(getHandle(seedNeighborhood));

    // Calculate mean for seed neighborhood
    af_array mean;
    af_array flatSeedNeighborhood;

    af_flat(&flatSeedNeighborhood, getHandle(seedNeighborhood));
    af_mean(&mean, flatSeedNeighborhood, 0);
    std::cout << "mean\n";
    af_print_array(mean);

    af_array variance;
    af_var(&variance, flatSeedNeighborhood, false, 0);

    af_array std_dev;
    af_sqrt(&std_dev, variance);
    std::cout << "std dev\n";
    af_print_array(std_dev);

    // const ArrayInfo& info = getInfo(std_dev);
    // af::dim4 dims  = info.dims();
    // af_dtype type  = info.getType();
    // std::cout << "type: " << type << std::endl;

    bool batchFlag = false;
    // \TODO Find thresholds:
    // lower = mean - mult * std_dev
    // upper = mean + mult * std_dev
    af_array product;
    auto test = getArray<float>(std_dev);

    auto multArgArray = createValueArray<unsigned>(test.dims(), mult);
    af_mul(&product, getHandle(multArgArray), std_dev, batchFlag);
    std::cout << "product\n";
    af_print_array(product);

    af_array lower_raw;
    af_sub(&lower_raw, mean, product, batchFlag);
    std::cout << "lower raw\n";
    af_print_array(lower_raw);

    af_array upper_raw;
    af_add(&upper_raw, mean, product, batchFlag);
    std::cout << "upper raw\n";
    af_print_array(upper_raw);

    // \TODO Adjust thresholds to include the seed intensity value
    // Do this later.

    return getHandle<T>(in);
    // // \TODO
    // int seedRow = 255;
    // int seedCol = 255;

    // return ConfidenceConnectedComponents(in, seedRow, seedCol, 1, 1);
}



af_err af_confidence_cc(af_array* out, const af_array in, const af_array seed,
                        const af_cc_type cc_type,
                        const unsigned radius, const unsigned multiplier, const int iter)
{
    try {
        const ArrayInfo& info = getInfo(in);

        af::dim4 dims = info.dims();
        ARG_ASSERT(2, (dims.ndims()>=0 && dims.ndims()<=3));

        af_array output;
        af_dtype type = info.getType();

        switch(type) {
            case u8: output = ccHelper<uchar>(getArray<uchar>(in), getArray<uchar>(seed), cc_type, radius, multiplier, iter); break;
            default : TYPE_ERROR(0, type);
        }

#if 0
        // \NOTE CUDA backend
        // ---
#include <af/seq.h>
#include <af/index.h>
#include <regions.hpp>
#include <copy.hpp>
#include <assign.hpp>
#include <math.hpp>
#include <tile.hpp>
        output = getHandle(detail::regions<uint>(castArray<char>(output), AF_CONNECTIVITY_8));

        std::vector<af_seq> seq(2, af_span);
        seq[0] = af_make_seq(0,11,1);

        auto blah = createSubArray<unsigned>(getArray<unsigned>(output),seq,false);
#endif

        // \TODO return region which covering the seed(s)

        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}


#if 0

    // extract values from seed Array
    std::vector<af_seq> index(4);

    for (int i = 0; i < 4; ++i) {
        index[i] = af_span;
    }
    af_seq s = {(double)0, 1, 1};
    index[0] = s;

    af_print_array(getHandle<T>(seed));
    Array<T> out = createSubArray(seed,index);
    af_print_array(getHandle<T>(out));

    // af_array idx;
    // // af_create_array(idx
    // float arr []  = { 0 };
    // dim_t dim[] = { 1 };
    // af_create_array(&idx, arr, 1, dim, s32);
    // std::cout << "idx\n";
    // af_print_array(idx);
    // std::cout << "hey\n";
    // af_print_array(getHandle(out));
    // std::cout << "there\n";


    // std::vector<af_seq> seqBegin(4, af_span);
    // seqBegin[0] = af_make_seq(0, 0, 1);

    // auto first = createSubArray(seed, seqBegin, false);
    // af_print_array(getHandle(first));

    // auto out = detail::lookup(in, first, 2);
    // af_print_array(getHandle(out));
#endif
