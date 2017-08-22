/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/image.h>
#include <vector>
#include <testHelpers.hpp>

using namespace af;
using std::vector;
using std::abs;

// \MAL taken from Regions test
template<typename T>
class ConfidenceConnected : public ::testing::Test
{
    public:
        virtual void SetUp() {}
};

// typedef ::testing::Types<float, int, uint, short, ushort, uchar, double> TestTypes;
typedef ::testing::Types<uchar> TestTypes; // \TODO

TYPED_TEST_CASE(ConfidenceConnected, TestTypes);

template<typename T>
void confidenceConnectedTest(std::string testFile, int iter)
{
    vector<af::dim4> numDims;
    vector<vector<T> > in;
    vector<vector<T> > out;

    readTests<T, T, int>(testFile, numDims, in, out);

    af::dim4 inDims = numDims[0];
    af::dim4 seedDims = numDims[1];
    af_array outArray = 0;
    af_array inArray = 0;
    af_array seedArray = 0;

    ASSERT_EQ(AF_SUCCESS, af_create_array(&inArray, &(in[0].front()),
                                          inDims.ndims(), inDims.get(), (af_dtype)af::dtype_traits<T>::af_type));
    ASSERT_EQ(AF_SUCCESS, af_create_array(&seedArray, &(in[1].front()),
                                          seedDims.ndims(), seedDims.get(), u8));
    af_cc_type ccType;
    ccType = AF_CC_CONFIDENCE;
    unsigned radius = 1;
    unsigned mult = 1;
    ASSERT_EQ(AF_SUCCESS, af_confidence_connected(&outArray, inArray, ccType, seedArray, radius, mult, iter));

    std::vector<T> outData(inDims.elements());

    ASSERT_EQ(AF_SUCCESS, af_get_data_ptr((void*)outData.data(), outArray));

    vector<T> goldOutput = out[0];
    size_t nElems = goldOutput.size();
    for (size_t i = 0; i < nElems; ++i) {
        ASSERT_EQ(goldOutput[i], outData[i]) << "at: " << i << std::endl;
    }

    ASSERT_EQ(AF_SUCCESS, af_release_array(inArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(seedArray));
    ASSERT_EQ(AF_SUCCESS, af_release_array(outArray));

}

#define CONF_CC_INIT(desc, file, iter)                                  \
    TYPED_TEST(ConfidenceConnected, desc)                               \
    {                                                                   \
        confidenceConnectedTest<TypeParam>(std::string(TEST_DIR "/connected_components/"#file"_"#iter".test"), iter); \
    }

    CONF_CC_INIT(ConfidenceConnected0, conf_cc_5x5, 1)
    CONF_CC_INIT(ConfidenceConnected1, conf_cc_5x5, 2)
    CONF_CC_INIT(ConfidenceConnected2, conf_cc_512x512, 1)

