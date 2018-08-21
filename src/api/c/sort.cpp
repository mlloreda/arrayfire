/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/algorithm.h>
#include <af/defines.h>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <sort.hpp>
#include <sort_index.hpp>
#include <sort_by_key.hpp>
#include <copy.hpp>

#include<cstdio>

using af::dim4;
using namespace detail;

template<typename T>
static inline af_array sort(const af_array in, const unsigned dim, const bool isAscending)
{
    const Array<T> &inArray = getArray<T>(in);
    return getHandle(sort<T>(inArray, dim, isAscending));
}

af_err af_sort(af_array *out, const af_array in, const unsigned dim, const bool isAscending)
{
    try {
        ARG_SETUP(in);

        if (in_info.elements() == 0) {
            return af_retain_array(out, in);
        }
        DIM_ASSERT(1, in_info.elements() > 0);

        af_array val;
        switch(in_info.getType()) {
            case f32: val = sort<float  >(in, dim, isAscending);  break;
            case f64: val = sort<double >(in, dim, isAscending);  break;
            case s32: val = sort<int    >(in, dim, isAscending);  break;
            case u32: val = sort<uint   >(in, dim, isAscending);  break;
            case s16: val = sort<short  >(in, dim, isAscending);  break;
            case u16: val = sort<ushort >(in, dim, isAscending);  break;
            case s64: val = sort<intl   >(in, dim, isAscending);  break;
            case u64: val = sort<uintl  >(in, dim, isAscending);  break;
            case u8:  val = sort<uchar  >(in, dim, isAscending);  break;
            case b8:  val = sort<char   >(in, dim, isAscending);  break;
            default:  TYPE_ERROR(in);
        }
        std::swap(*out, val);
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<typename T>
static inline void sort_index(af_array *val, af_array *idx, const af_array in,
                              const unsigned dim, const bool isAscending)
{
    const Array<T> &inArray = getArray<T>(in);

    // Initialize Dummy Arrays
    Array<T> valArray = createEmptyArray<T>(af::dim4());
    Array<uint> idxArray = createEmptyArray<uint>(af::dim4());

    sort_index<T>(valArray, idxArray, inArray, dim, isAscending);
    *val = getHandle(valArray);
    *idx = getHandle(idxArray);
}

af_err af_sort_index(af_array *out, af_array *indices, const af_array in, const unsigned dim, const bool isAscending)
{
    try {
        ARG_SETUP(in);

        if (in_info.elements() <= 0) {
            const af_dtype in_type = in_info.getType();
            AF_CHECK(af_create_handle(out,     0, nullptr, in_type));
            AF_CHECK(af_create_handle(indices, 0, nullptr, in_type));
            return AF_SUCCESS;
        }

        af_array val;
        af_array idx;
        switch(in_info.getType()) {
            case f32: sort_index<float  >(&val, &idx, in, dim, isAscending);  break;
            case f64: sort_index<double >(&val, &idx, in, dim, isAscending);  break;
            case s32: sort_index<int    >(&val, &idx, in, dim, isAscending);  break;
            case u32: sort_index<uint   >(&val, &idx, in, dim, isAscending);  break;
            case s16: sort_index<short  >(&val, &idx, in, dim, isAscending);  break;
            case u16: sort_index<ushort >(&val, &idx, in, dim, isAscending);  break;
            case s64: sort_index<intl   >(&val, &idx, in, dim, isAscending);  break;
            case u64: sort_index<uintl  >(&val, &idx, in, dim, isAscending);  break;
            case u8:  sort_index<uchar  >(&val, &idx, in, dim, isAscending);  break;
            case b8:  sort_index<char   >(&val, &idx, in, dim, isAscending);  break;
            default:  TYPE_ERROR(in);
        }
        std::swap(*out, val);
        std::swap(*indices, idx);
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<typename Tk, typename Tv>
static inline void sort_by_key(af_array *okey, af_array *oval, const af_array ikey, const af_array ival,
                               const unsigned dim, const bool isAscending)
{
    const Array<Tk> &ikeyArray = getArray<Tk>(ikey);
    const Array<Tv> &ivalArray = getArray<Tv>(ival);

    // Initialize Dummy Arrays
    Array<Tk> okeyArray = createEmptyArray<Tk>(af::dim4());
    Array<Tv> ovalArray = createEmptyArray<Tv>(af::dim4());

    sort_by_key<Tk, Tv>(okeyArray, ovalArray, ikeyArray, ivalArray, dim, isAscending);
    *okey = getHandle(okeyArray);
    *oval = getHandle(ovalArray);
}

template<typename Tk>
void sort_by_key_tmplt(af_array *okey, af_array *oval, const af_array ikey, const af_array ival,
                       const unsigned dim, const bool isAscending)
{
    ARG_SETUP(ival);

    switch(ival_info.getType()) {
    case f32: sort_by_key<Tk, float  >(okey, oval, ikey, ival, dim, isAscending);  break;
    case f64: sort_by_key<Tk, double >(okey, oval, ikey, ival, dim, isAscending);  break;
    case c32: sort_by_key<Tk, cfloat >(okey, oval, ikey, ival, dim, isAscending);  break;
    case c64: sort_by_key<Tk, cdouble>(okey, oval, ikey, ival, dim, isAscending);  break;
    case s32: sort_by_key<Tk, int    >(okey, oval, ikey, ival, dim, isAscending);  break;
    case u32: sort_by_key<Tk, uint   >(okey, oval, ikey, ival, dim, isAscending);  break;
    case s16: sort_by_key<Tk, short  >(okey, oval, ikey, ival, dim, isAscending);  break;
    case u16: sort_by_key<Tk, ushort >(okey, oval, ikey, ival, dim, isAscending);  break;
    case s64: sort_by_key<Tk, intl   >(okey, oval, ikey, ival, dim, isAscending);  break;
    case u64: sort_by_key<Tk, uintl  >(okey, oval, ikey, ival, dim, isAscending);  break;
    case u8:  sort_by_key<Tk, uchar  >(okey, oval, ikey, ival, dim, isAscending);  break;
    case b8:  sort_by_key<Tk, char   >(okey, oval, ikey, ival, dim, isAscending);  break;
    default:  TYPE_ERROR(ival);
    }

    return;
}

af_err af_sort_by_key(af_array *out_keys, af_array *out_values,
                      const af_array keys, const af_array values,
                      const unsigned dim, const bool isAscending)
{
    try {
        ARG_SETUP(keys);
        ARG_SETUP(values);
        ASSERT_DIM(keys, values);
        TYPE_ASSERT(keys_info.isReal()); // \TODO(miguel)
        if (keys_info.elements() == 0) {
            AF_CHECK(af_create_handle(out_keys,   0, nullptr, keys_info.getType()));
            AF_CHECK(af_create_handle(out_values, 0, nullptr, keys_info.getType()));
            return AF_SUCCESS;
        }

        af_array oKey;
        af_array oVal;
        switch(keys_info.getType()) {
            case f32: sort_by_key_tmplt<float  >(&oKey, &oVal, keys, values, dim, isAscending);  break;
            case f64: sort_by_key_tmplt<double >(&oKey, &oVal, keys, values, dim, isAscending);  break;
            case s32: sort_by_key_tmplt<int    >(&oKey, &oVal, keys, values, dim, isAscending);  break;
            case u32: sort_by_key_tmplt<uint   >(&oKey, &oVal, keys, values, dim, isAscending);  break;
            case s16: sort_by_key_tmplt<short  >(&oKey, &oVal, keys, values, dim, isAscending);  break;
            case u16: sort_by_key_tmplt<ushort >(&oKey, &oVal, keys, values, dim, isAscending);  break;
            case s64: sort_by_key_tmplt<intl   >(&oKey, &oVal, keys, values, dim, isAscending);  break;
            case u64: sort_by_key_tmplt<uintl  >(&oKey, &oVal, keys, values, dim, isAscending);  break;
            case u8:  sort_by_key_tmplt<uchar  >(&oKey, &oVal, keys, values, dim, isAscending);  break;
            case b8:  sort_by_key_tmplt<char   >(&oKey, &oVal, keys, values, dim, isAscending);  break;
            default:  TYPE_ERROR(keys);
        }
        std::swap(*out_keys, oKey);
        std::swap(*out_values, oVal);
    }
    CATCHALL;

    return AF_SUCCESS;
}
