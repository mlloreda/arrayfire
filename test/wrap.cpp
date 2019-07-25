/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#define GTEST_LINKED_AS_SHARED_LIBRARY 1
#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <algorithm>
#include <complex>
#include <iostream>
#include <string>
#include <vector>

using af::allTrue;
using af::array;
using af::cdouble;
using af::cfloat;
using af::dim4;
using af::dtype;
using af::dtype_traits;
using af::randu;
using af::range;
using std::abs;
using std::cout;
using std::endl;
using std::string;
using std::vector;

template<typename T>
class Wrap : public ::testing::Test {
   public:
    virtual void SetUp() {}
};

// create a list of types to be tested
typedef ::testing::Types<float, double, cfloat, cdouble, int, unsigned int,
                         intl, uintl, char, unsigned char, short, ushort>
    TestTypes;

// register the type list
TYPED_TEST_CASE(Wrap, TestTypes);

template<typename T>
inline double get_val(T val) {
    return val;
}

template<>
inline double get_val<cfloat>(cfloat val) {
    return abs(val);
}

template<>
inline double get_val<cdouble>(cdouble val) {
    return abs(val);
}

template<>
inline double get_val<unsigned char>(unsigned char val) {
    return ((int)(val)) % 256;
}

template<>
inline double get_val<char>(char val) {
    return (val != 0);
}

template<typename T>
void wrapTest(const dim_t ix, const dim_t iy, const dim_t wx, const dim_t wy,
              const dim_t sx, const dim_t sy, const dim_t px, const dim_t py,
              bool cond) {
    SUPPORTED_TYPE_CHECK(T);

    const int nc = 1;

    int lim = std::max((dim_t)2, (dim_t)(250) / (wx * wy));

    dtype ty = (dtype)dtype_traits<T>::af_type;
    array in = round(lim * randu(ix, iy, nc, f32)).as(ty);

    vector<T> h_in(in.elements());
    in.host(&h_in[0]);

    vector<int> h_factor(ix * iy);

    dim_t ny = (iy + 2 * py - wy) / sy + 1;
    dim_t nx = (ix + 2 * px - wx) / sx + 1;

    for (int idy = 0; idy < ny; idy++) {
        int fy = idy * sy - py;
        if (fy + wy < 0 || fy >= iy) continue;

        for (int idx = 0; idx < nx; idx++) {
            int fx = idx * sx - px;
            if (fx + wx < 0 || fx >= ix) continue;

            for (int ly = 0; ly < wy; ly++) {
                if (fy + ly < 0 || fy + ly >= iy) continue;

                for (int lx = 0; lx < wx; lx++) {
                    if (fx + lx < 0 || fx + lx >= ix) continue;
                    h_factor[(fy + ly) * ix + (fx + lx)]++;
                }
            }
        }
    }

    array factor(ix, iy, &h_factor[0]);

    array in_dim  = unwrap(in, wx, wy, sx, sy, px, py, cond);
    array res_dim = wrap(in_dim, ix, iy, wx, wy, sx, sy, px, py, cond);

    ASSERT_EQ(in.elements(), res_dim.elements());

    vector<T> h_res(ix * iy);
    res_dim.host(&h_res[0]);

    for (int n = 0; n < nc; n++) {
        T *iptr = &h_in[n * ix * iy];
        T *rptr = &h_res[n * ix * iy];

        for (int y = 0; y < iy; y++) {
            for (int x = 0; x < ix; x++) {
                // FIXME: Use a better test
                T ival     = iptr[y * ix + x];
                T rval     = rptr[y * ix + x];
                int factor = h_factor[y * ix + x];

                if (get_val(ival) == 0) continue;

                ASSERT_NEAR(get_val<T>(ival * factor), get_val<T>(rval), 1E-5)
                    << "at " << x << "," << y << " for cond  == " << cond
                    << endl;
            }
        }
    }
}

#define WRAP_INIT(desc, ix, iy, wx, wy, sx, sy, px, py)             \
    TYPED_TEST(Wrap, Col##desc) {                                   \
        wrapTest<TypeParam>(ix, iy, wx, wy, sx, sy, px, py, true);  \
    }                                                               \
    TYPED_TEST(Wrap, Row##desc) {                                   \
        wrapTest<TypeParam>(ix, iy, wx, wy, sx, sy, px, py, false); \
    }

WRAP_INIT(00, 300, 100, 3, 3, 1, 1, 0, 0);
WRAP_INIT(01, 300, 100, 3, 3, 1, 1, 1, 1);
WRAP_INIT(03, 300, 100, 3, 3, 2, 2, 0, 0);
WRAP_INIT(04, 300, 100, 3, 3, 2, 2, 1, 1);
WRAP_INIT(05, 300, 100, 3, 3, 2, 2, 2, 2);
WRAP_INIT(06, 300, 100, 3, 3, 3, 3, 0, 0);
WRAP_INIT(07, 300, 100, 3, 3, 3, 3, 1, 1);
WRAP_INIT(08, 300, 100, 3, 3, 3, 3, 2, 2);
WRAP_INIT(09, 300, 100, 4, 4, 1, 1, 0, 0);
WRAP_INIT(13, 300, 100, 4, 4, 2, 2, 0, 0);
WRAP_INIT(14, 300, 100, 4, 4, 2, 2, 1, 1);
WRAP_INIT(15, 300, 100, 4, 4, 2, 2, 2, 2);
WRAP_INIT(16, 300, 100, 4, 4, 2, 2, 3, 3);
WRAP_INIT(17, 300, 100, 4, 4, 4, 4, 0, 0);
WRAP_INIT(18, 300, 100, 4, 4, 4, 4, 1, 1);
WRAP_INIT(19, 300, 100, 4, 4, 4, 4, 2, 2);
WRAP_INIT(27, 300, 100, 8, 8, 8, 8, 0, 0);
WRAP_INIT(28, 300, 100, 8, 8, 8, 8, 7, 7);
WRAP_INIT(31, 300, 100, 12, 12, 12, 12, 0, 0);
WRAP_INIT(32, 300, 100, 12, 12, 12, 12, 2, 2);
WRAP_INIT(35, 300, 100, 16, 16, 16, 16, 15, 15);
WRAP_INIT(36, 300, 100, 31, 31, 8, 8, 15, 15);
WRAP_INIT(39, 300, 100, 8, 12, 8, 12, 0, 0);
WRAP_INIT(40, 300, 100, 8, 12, 8, 12, 7, 11);
WRAP_INIT(43, 300, 100, 15, 10, 15, 10, 0, 0);
WRAP_INIT(44, 300, 100, 15, 10, 15, 10, 14, 9);

TEST(Wrap, MaxDim) {
    const size_t largeDim = 65535 + 1;
    array input           = range(5, 5, 1, largeDim);

    const unsigned wx = 5;
    const unsigned wy = 5;
    const unsigned sx = 5;
    const unsigned sy = 5;
    const unsigned px = 0;
    const unsigned py = 0;

    array unwrapped = unwrap(input, wx, wy, sx, sy, px, py);
    array output    = wrap(unwrapped, 5, 5, wx, wy, sx, sy, px, py);

    ASSERT_ARRAYS_EQ(output, input);
}

TEST(Wrap, DocSnippet) {
    //! [ex_wrap_1]
    float hA[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    array A(dim4(3, 3), hA);
    //  1.     4.     7.
    //  2.     5.     8.
    //  3.     6.     9.

    array A_unwrapped = unwrap(A, 2, 2,  // window size
                               2, 2,     // stride (distinct)
                               1, 1);    // padding
    //  0.     0.     0.     5.
    //  0.     0.     4.     6.
    //  0.     2.     0.     8.
    //  1.     3.     7.     9.

    array A_wrapped = wrap(A_unwrapped, 3, 3,  // A's size
                           2, 2,               // window size
                           2, 2,               // stride (distinct)
                           1, 1);              // padding
    //  1.     4.     7.
    //  2.     5.     8.
    //  3.     6.     9.
    //! [ex_wrap_1]

    ASSERT_ARRAYS_EQ(A, A_wrapped);

    //! [ex_wrap_2]
    float hB[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    array B(dim4(3, 3), hB);
    //  1.     1.     1.
    //  1.     1.     1.
    //  1.     1.     1.
    array B_unwrapped = unwrap(B, 2, 2,  // window size
                               1, 1);    // stride (sliding)
    //  1.     1.     1.     1.
    //  1.     1.     1.     1.
    //  1.     1.     1.     1.
    //  1.     1.     1.     1.
    array B_wrapped = wrap(B_unwrapped, 3, 3,  // B's size
                           2, 2,               // window size
                           1, 1);              // stride (sliding)
    //  1.     2.     1.
    //  2.     4.     2.
    //  1.     2.     1.
    //! [ex_wrap_2]

    float gold_hB_wrapped[] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
    array gold_B_wrapped(dim4(3, 3), gold_hB_wrapped);
    ASSERT_ARRAYS_EQ(gold_B_wrapped, B_wrapped);
}

static void getInput(af_array *data, const dim_t *dims) {
    float h_data[16] = { 10, 20, 20, 30,
                         30, 40, 40, 50,
                         30, 40, 40, 50,
                         50, 60, 60, 70 };
    ASSERT_SUCCESS(af_create_array(data, &h_data[0], 2, dims, f32));
}
static void getGold(af_array *gold, const dim_t *dims) {
    float h_gold[16]= { 10, 20, 30, 40,
                        20, 30, 40, 50,
                        30, 40, 50, 60,
                        40, 50, 60, 70 };
    ASSERT_SUCCESS(af_create_array(gold, &h_gold[0], 2, dims, f32));
}

template<typename T>
class WrapV2 : public ::testing::Test {};

TYPED_TEST_CASE(WrapV2, TestTypes);

template<typename T>
void testSpclOutArray(float *h_gold, dim4 gold_dims, float *h_in, dim4 in_dims,
                      TestOutputArrayType out_array_type) {
    SUPPORTED_TYPE_CHECK(T);
    typedef typename dtype_traits<T>::base_type BT;

    vector<T> h_gold_cast(gold_dims.elements());
    vector<T> h_in_cast(in_dims.elements());

    for (int i = 0; i < gold_dims.elements(); ++i) {
        h_gold_cast[i] = static_cast<T>(h_gold[i]);
    }
    for (int i = 0; i < in_dims.elements(); ++i) {
        h_in_cast[i] = static_cast<T>(h_in[i]);
    }

    af_array in  = 0;
    af_array pos = 0;

    ASSERT_SUCCESS(af_create_array(&in, &h_in_cast.front(), in_dims.ndims(),
                                   in_dims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    af_array out = 0;
    TestOutputArrayInfo metadata(out_array_type);
    genTestOutputArray(&out, gold_dims.ndims(), gold_dims.get(),
                       (af_dtype)dtype_traits<T>::af_type, &metadata);
    // Taken from the Wrap.DocSnippet test
    ASSERT_SUCCESS(af_wrap_v2(&out, in, 3, 3,  // output dims
                              2, 2,            // window size
                              2, 2,            // stride
                              1, 1,            // padding
                              true));          // is_column

    af_array gold = 0;
    ASSERT_SUCCESS(af_create_array(&gold, &h_gold_cast.front(),
                                   gold_dims.ndims(), gold_dims.get(),
                                   (af_dtype)dtype_traits<T>::af_type));

    ASSERT_SPECIAL_ARRAYS_EQ(gold, out, &metadata);

    if (gold != 0) { ASSERT_SUCCESS(af_release_array(gold)); }
    if (in != 0) { ASSERT_SUCCESS(af_release_array(in)); }
}

TYPED_TEST(WrapV2, UseNullOutputArray) {
    // Taken from Wrap.DocSnippet test
    float h_in[16] = {0.0f, 0.0f, 0.0f, 1.0f,
                      0.0f, 0.0f, 2.0f, 3.0f,
                      0.0f, 4.0f, 0.0f, 7.0f,
                      5.0f, 6.0f, 8.0f, 9.0f};
    dim4 in_dims(4, 4);

    float h_gold[9] = {1.0f, 2.0f, 3.0f,
                       4.0f, 5.0f, 6.0f,
                       7.0f, 8.0f, 9.0f};
    dim4 gold_dims(3, 3);

    SCOPED_TRACE("UseNullOutputArray");
    testSpclOutArray<TypeParam>(h_gold, gold_dims, h_in, in_dims, NULL_ARRAY);
}

TYPED_TEST(WrapV2, UseFullExistingOutputArray) {
    // Taken from Wrap.DocSnippet test
    float h_in[16] = {0.0f, 0.0f, 0.0f, 1.0f,
                      0.0f, 0.0f, 2.0f, 3.0f,
                      0.0f, 4.0f, 0.0f, 7.0f,
                      5.0f, 6.0f, 8.0f, 9.0f};
    dim4 in_dims(4, 4);

    float h_gold[9] = {1.0f, 2.0f, 3.0f,
                       4.0f, 5.0f, 6.0f,
                       7.0f, 8.0f, 9.0f};
    dim4 gold_dims(3, 3);

    SCOPED_TRACE("UseFullExistingOutputArray");
    testSpclOutArray<TypeParam>(h_gold, gold_dims, h_in, in_dims, FULL_ARRAY);
}

TYPED_TEST(WrapV2, UseExistingOutputSubArray) {
    // Taken from Wrap.DocSnippet test
    float h_in[16] = {0.0f, 0.0f, 0.0f, 1.0f,
                      0.0f, 0.0f, 2.0f, 3.0f,
                      0.0f, 4.0f, 0.0f, 7.0f,
                      5.0f, 6.0f, 8.0f, 9.0f};
    dim4 in_dims(4, 4);

    float h_gold[9] = {1.0f, 2.0f, 3.0f,
                       4.0f, 5.0f, 6.0f,
                       7.0f, 8.0f, 9.0f};
    dim4 gold_dims(3, 3);

    SCOPED_TRACE("UseExistingOutputSubArray");
    testSpclOutArray<TypeParam>(h_gold, gold_dims, h_in, in_dims, SUB_ARRAY);
}

TYPED_TEST(WrapV2, UseReorderedOutputArray) {
    // Taken from Wrap.DocSnippet test
    float h_in[16] = {0.0f, 0.0f, 0.0f, 1.0f,
                      0.0f, 0.0f, 2.0f, 3.0f,
                      0.0f, 4.0f, 0.0f, 7.0f,
                      5.0f, 6.0f, 8.0f, 9.0f};
    dim4 in_dims(4, 4);

    float h_gold[9] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    dim4 gold_dims(3, 3);

    SCOPED_TRACE("UseReorderedOutputArray");
    testSpclOutArray<TypeParam>(h_gold, gold_dims, h_in, in_dims,
                                REORDERED_ARRAY);
}

class WrapSimple : virtual public ::testing::Test {
   protected:
    virtual void SetUp() {
        in_dims[0] = 4;
        in_dims[1] = 4;
        in_dims[2] = 1;
        in_dims[3] = 1;

        ::getInput(&in_, &in_dims[0]);
        ::getGold(&gold_, &in_dims[0]);

        win_len   = 2;
        strd_len  = 2;
        pad_len   = 0;
        is_column = true;
    }
    virtual void TearDown() {
        if (in_ != 0) af_release_array(in_);
        if (gold_ != 0) af_release_array(gold_);
    }

    af_array in_;
    af_array gold_;
    dim_t in_dims[4];
    dim_t win_len;
    dim_t strd_len;
    dim_t pad_len;
    bool is_column;
};

class ArgDim {
   public:
    ArgDim(dim_t d0, dim_t d1) : dim0(d0), dim1(d1) {}
    void get(dim_t *d0, dim_t *d1);

   private:
    dim_t dim0;
    dim_t dim1;
};
void ArgDim::get(dim_t *d0, dim_t *d1) {
    *d0 = this->dim0;
    *d1 = this->dim1;
}
class WindowDims : public ArgDim {
   public:
    WindowDims() : ArgDim(1, 1) {}
    WindowDims(dim_t d0, dim_t d1) : ArgDim(d0, d1) {}
};
class StrideDims : public ArgDim {
   public:
    StrideDims() : ArgDim(1, 1) {}
    StrideDims(dim_t d0, dim_t d1) : ArgDim(d0, d1) {}
};
class PadDims : public ArgDim {
   public:
    PadDims() : ArgDim(0, 0) {}
    PadDims(dim_t d0, dim_t d1) : ArgDim(d0, d1) {}
};

class WrapArgs {
   public:
    WindowDims *wc_;
    StrideDims *sc_;
    PadDims *pc_;
    bool is_column;
    af_err err;
    WrapArgs(dim_t win_d0, dim_t win_d1, dim_t str_d0, dim_t str_d1,
             dim_t pad_d0, dim_t pad_d1, bool is_col, af_err err)
        : wc_(new WindowDims(win_d0, win_d1))
        , sc_(new StrideDims(str_d0, str_d1))
        , pc_(new PadDims(pad_d0, pad_d1))
        , is_column(is_col)
        , err(err) {}
    WrapArgs()
        : wc_(new WindowDims())
        , sc_(new StrideDims())
        , pc_(new PadDims())
        , is_column(true)
        , err(af_err(999)) {}
};

class WrapAPITest
    : public WrapSimple
    , public ::testing::WithParamInterface<WrapArgs> {
   public:
    virtual void SetUp() {
        in_dims[0] = 4;
        in_dims[1] = 4;
        in_dims[2] = 1;
        in_dims[3] = 1;

        input = GetParam();
        ::getInput(&in_, &in_dims[0]);
    }
    virtual void TearDown() {
        if (in_ != 0) af_release_array(in_);
    }

    WrapArgs input;
    af_array in_;
    dim_t in_dims[4];
};
TEST_P(WrapAPITest, CheckDifferentWrapArgs) {
    WindowDims *wc = input.wc_;
    StrideDims *sc = input.sc_;
    PadDims *pc    = input.pc_;

    dim_t win_d0, win_d1;
    dim_t str_d0, str_d1;
    dim_t pad_d0, pad_d1;
    wc->get(&win_d0, &win_d1);
    sc->get(&str_d0, &str_d1);
    pc->get(&pad_d0, &pad_d1);

    af_array out_ = 0;
    af_err err    = af_wrap(&out_, in_, in_dims[0], in_dims[1], win_d0, win_d1,
                         str_d0, str_d1, pad_d0, pad_d1, input.is_column);

    ASSERT_EQ(err, input.err);
    if (out_ != 0) af_release_array(out_);
}

WrapArgs args[] = {
    // clang-format off
    //      | win_dim0 | win_dim1 | str_dim0 | str_dim1 | pad_dim0 | pad_dim1 | is_col |    err    |
    WrapArgs(        2,         2,         2,         2,         0,         0,    true,  AF_SUCCESS),
    WrapArgs(        2,         2,         2,         2,         0,         0,   false,  AF_SUCCESS),

    WrapArgs(       -1,         2,         2,         2,         0,         0,    true,  AF_ERR_ARG),
    WrapArgs(        2,        -1,         2,         2,         0,         0,    true,  AF_ERR_ARG),
    WrapArgs(       -1,        -1,         2,         2,         0,         0,    true,  AF_ERR_ARG),

    WrapArgs(        2,         2,        -1,         2,         0,         0,    true,  AF_ERR_ARG),
    WrapArgs(        2,         2,         2,        -1,         0,         0,    true,  AF_ERR_ARG),
    WrapArgs(        2,         2,        -1,        -1,         0,         0,    true,  AF_ERR_ARG),

    WrapArgs(        2,         2,         2,         2,         1,         1,    true,  AF_ERR_SIZE),
    WrapArgs(        2,         2,         2,         2,        -1,         1,    true,  AF_ERR_SIZE),
    WrapArgs(        2,         2,         2,         2,         1,        -1,    true,  AF_ERR_SIZE),
    WrapArgs(        2,         2,         2,         2,        -1,        -1,    true,  AF_ERR_SIZE),
    // clang-format on
};
INSTANTIATE_TEST_CASE_P(BulkTest, WrapAPITest, ::testing::ValuesIn(args));
