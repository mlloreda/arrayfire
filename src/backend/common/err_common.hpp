/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <af/defines.h>
#include <common/defines.hpp>
#include "ArrayInfo.hpp"

#include <cassert>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

class AfError   : public std::logic_error
{
    std::string functionName;
    std::string fileName;
    int lineNumber;
    af_err error;
    AfError();

public:

    AfError(const char * const func,
            const char * const file,
            const int line,
            const char * const message, af_err err);

    AfError(std::string func,
            std::string file,
            const int line,
            std::string message, af_err err);

    const std::string&
    getFunctionName() const;

    const std::string&
    getFileName() const;

    int getLine() const;

    af_err getError() const;

    virtual ~AfError() throw();
};

// TODO: Perhaps add a way to return supported types
class TypeError : public AfError
{
    int argIndex;
    std::string errTypeName;
    TypeError();

public:

    TypeError(const char * const func,
              const char * const file,
              const int line,
              const int index,
              const af_dtype type);

    const std::string&
    getTypeName() const;

    int getArgIndex() const;

    ~TypeError() throw() {}
};

class ArgumentError : public AfError
{
    int argIndex;
    std::string expected;
    ArgumentError();

public:

    ArgumentError(const char * const func,
                  const char * const file,
                  const int line,
                  const int index,
                  const char * const expectString);

    const std::string&
    getExpectedCondition() const;

    int getArgIndex() const;

    ~ArgumentError() throw(){}
};

class SupportError  :   public AfError
{
    std::string backend;
    SupportError();

public:

    SupportError(const char * const func,
                 const char * const file,
                 const int line,
                 const char * const back);

    ~SupportError()throw() {}

    const std::string&
    getBackendName() const;
};

class DimensionError : public AfError
{
    int argIndex;
    std::string expected;
    DimensionError();

public:

    DimensionError(const char * const func,
                   const char * const file,
                   const int line,
                   const int index,
                   const char * const expectString);

    const std::string&
    getExpectedCondition() const;

    int getArgIndex() const;

    ~DimensionError() throw(){}
};

af_err processException();

std::string getEnumString(const af_dtype type);

void print_error(const std::string &msg);

////////////////
// Type asserts
////////////////

// ASSERT_TYPE(ARR, vector<af_dtype>)
void assert_type(const af_dtype in_type, const char * in_str,
                const std::vector<af_dtype> types,
                const char * file, const char * function, const int line);

// ASSERT_TYPE_EQ(af_array, af_array)
void assert_type_eq(const af_dtype lhs_type, const af_dtype rhs_type,
              const char * lhs_str, const char * rhs_str,
              const char * file, const char * function, const int line);

// TYPE_ERROR(af_array)
void type_error(const ArrayInfo &in_info, const char * in_str,
                const char * file, const char * function, const int line);

// UNSUPPORTED_TYPE(af_dtype)
void unsupported_type(const af_dtype in, const char * in_str,
                      const char * file, const char * function, const int line);

///////////////
// Dim asserts
///////////////

// ASSERT_DIM(af_array, af_array)
void dim_eq(const ArrayInfo &lhs, const ArrayInfo &rhs,
            const char * lhs_str, const char * rhs_str,
            const char * file, const char * function, const int line);

// ASSERT_DIM_{LT,GT,EQ}(dim_idx, dim_val, af_array)
void dim_cmp(const int op,
             const int dim_idx, const int dim_val, const ArrayInfo &arr_info,
             const char * arr_str,
             const char * file, const char * function, const int line);

#define ARG_SETUP(arg) \
     const ArrayInfo& arg##_info = getInfo(arg)

#define TYPES(...) __VA_ARGS__

// Types
#define ASSERT_TYPE(ARR, ...) do {                                  \
        assert_type(getInfo(ARR).getType(), #ARR, {__VA_ARGS__},    \
                    __FILE__, __FUNCTION__, __LINE__);              \
    } while(0)
#define ASSERT_TYPE_EQ(LHS, RHS) do {                                   \
        assert_type_eq(getInfo(LHS).getType(), getInfo(RHS).getType(),  \
                       #LHS, #RHS,                                      \
                       __FILE__, __FUNCTION__, __LINE__);               \
    } while(0)
#define TYPE_ERROR(...) do {                            \
        type_error(getInfo(__VA_ARGS__), #__VA_ARGS__,  \
                   __FILE__, __FUNCTION__, __LINE__);   \
    } while(0)
#define UNSUPPORTED_TYPE(...) do {                          \
        unsupported_type(__VA_ARGS__, #__VA_ARGS__,         \
                         __FILE__, __FUNCTION__, __LINE__); \
    } while (0)

// Dims
#define ASSERT_DIM(LHS, RHS) do {                   \
        dim_eq(getInfo(LHS), getInfo(RHS),          \
               #LHS, #RHS,                          \
               __FILE__, __FUNCTION__, __LINE__);   \
    } while (0)
#define ASSERT_DIM_LT(ARR, DIM_IDX, DIM_VAL) do {           \
        dim_cmp(0, DIM_IDX, DIM_VAL, getInfo(ARR), #ARR,    \
                __FILE__, __FUNCTION__, __LINE__);          \
    } while(0)
#define ASSERT_DIM_EQ(ARR, DIM_IDX, DIM_VAL) do {           \
        dim_cmp(1, DIM_IDX, DIM_VAL, getInfo(ARR), #ARR,    \
                __FILE__, __FUNCTION__, __LINE__);          \
    } while(0)
#define ASSERT_DIM_GT(ARR, DIM_IDX, DIM_VAL) do {           \
        dim_cmp(2, DIM_IDX, DIM_VAL, getInfo(ARR), #ARR,    \
                __FILE__, __FUNCTION__, __LINE__);          \
    } while(0)


#define DIM_ASSERT(INDEX, COND) do {                        \
        if((COND) == false) {                               \
            throw DimensionError(__PRETTY_FUNCTION__,       \
                                 __AF_FILENAME__, __LINE__, \
                                 INDEX, #COND);             \
        }                                                   \
    } while(0)

#define ARG_ASSERT(INDEX, COND) do {                        \
        if((COND) == false) {                               \
            throw ArgumentError(__PRETTY_FUNCTION__,        \
                                __AF_FILENAME__, __LINE__,  \
                                INDEX, #COND);              \
        }                                                   \
    } while(0)

#define AF_ERROR(MSG, ERR_TYPE) do {                        \
        throw AfError(__PRETTY_FUNCTION__,                  \
                      __AF_FILENAME__, __LINE__,            \
                      MSG, ERR_TYPE);                       \
    } while(0)

#define AF_RETURN_ERROR(MSG, ERR_TYPE) do {                 \
        AfError err(__PRETTY_FUNCTION__,                    \
                    __AF_FILENAME__, __LINE__,              \
                    MSG, ERR_TYPE);                         \
        std::stringstream s;                                \
        s << "Error in " << err.getFunctionName() << "\n"   \
          << "In file " << err.getFileName()                \
          << ":" << err.getLine()  << "\n"                  \
          << err.what() << "\n";                            \
        print_error(s.str());                               \
        return ERR_TYPE;                                    \
    } while(0)

#define TYPE_ASSERT(COND) do {                              \
        if ((COND) == false) {                              \
            AF_ERROR("Type mismatch inputs",                \
                     AF_ERR_DIFF_TYPE);                     \
        }                                                   \
    } while(0)

#define AF_ASSERT(COND, MESSAGE)                            \
    assert(MESSAGE && COND)

#define CATCHALL                                            \
    catch(...) {                                            \
        return processException();                          \
    }

#define AF_CHECK(fn) do {                                   \
        af_err __err = fn;                                  \
        if (__err == AF_SUCCESS) break;                     \
        throw AfError(__PRETTY_FUNCTION__,                  \
                      __AF_FILENAME__, __LINE__,            \
                      "\n", __err);                         \
    } while(0)

static const int MAX_ERR_SIZE = 1024;
std::string& get_global_error_string();
