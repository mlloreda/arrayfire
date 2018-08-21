/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/device.h>
#include <af/exception.h>
#include <common/err_common.hpp>
#include <common/util.hpp>
#include <type_util.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>

#if defined(WITH_GRAPHICS) && !defined(AF_UNIFIED)
#include <common/graphics_common.hpp>
#endif

#ifdef AF_OPENCL
#include <platform.hpp>
#include <errorcodes.hpp>
#endif

#include <spdlog/fmt/ostr.h>

using std::string;
using std::stringstream;
using std::vector;

AfError::AfError(const char * const func,
                 const char * const file,
                 const int line,
                 const char * const message, af_err err)
    : logic_error   (message),
      functionName  (func),
      fileName      (file),
      lineNumber(line),
      error(err)
{}

AfError::AfError(string func,
                 string file,
                 const int line,
                 string message, af_err err)
    : logic_error   (message),
      functionName  (func),
      fileName      (file),
      lineNumber(line),
      error(err)
{}

const string&
AfError::getFunctionName() const
{
    return functionName;
}

const string&
AfError::getFileName() const
{
    return fileName;
}

int
AfError::getLine() const
{
    return lineNumber;
}

af_err
AfError::getError() const
{
    return error;
}

AfError::~AfError() throw() {}

TypeError::TypeError(const char * const func,
                     const char * const file,
                     const int line,
                     const int index, const af_dtype type)
    : AfError (func, file, line, "Invalid data type", AF_ERR_TYPE),
      argIndex(index),
      errTypeName(getName(type))
{}

const string& TypeError::getTypeName() const
{
    return errTypeName;
}

int TypeError::getArgIndex() const
{
    return argIndex;
}

ArgumentError::ArgumentError(const char * const func,
                             const char * const file,
                             const int line,
                             const int index,
                             const char * const  expectString)
    : AfError(func, file, line, "Invalid argument", AF_ERR_ARG),
      argIndex(index),
      expected(expectString)
{

}

const string& ArgumentError::getExpectedCondition() const
{
    return expected;
}

int ArgumentError::getArgIndex() const
{
    return argIndex;
}


SupportError::SupportError(const char * const func,
                           const char * const file,
                           const int line,
                           const char * const back)
    : AfError(func, file, line, "Unsupported Error", AF_ERR_NOT_SUPPORTED),
      backend(back)
{}

const string& SupportError::getBackendName() const
{
    return backend;
}

DimensionError::DimensionError(const char * const  func,
                               const char * const file,
                               const int line,
                               const int index,
                               const char * const  expectString)
    : AfError(func, file, line, "Invalid size", AF_ERR_SIZE),
      argIndex(index),
      expected(expectString)
{

}

const string& DimensionError::getExpectedCondition() const
{
    return expected;
}

int DimensionError::getArgIndex() const
{
    return argIndex;
}

string getEnumString(const af_dtype type) {
    std::string name;
    switch (type) {
    case f32: name = "f32"; break;
    case c32: name = "c32"; break;
    case f64: name = "f64"; break;
    case c64: name = "c64"; break;
    case b8:  name = "b8";  break;
    case s32: name = "s32"; break;
    case u32: name = "u32"; break;
    case u8:  name = "u8";  break;
    case s64: name = "s64"; break;
    case u64: name = "u64"; break;
    case s16: name = "s16"; break;
    case u16: name = "u16"; break;
    default: assert(false && "Invalid type");
    }
    return name;
}

void
assert_type(const af_dtype in_type, const char * in_str,
            const vector<af_dtype> types,
            const char * file, const char * function, const int line)
{
    vector<af_dtype>::const_iterator it;
    it = std::find(std::begin(types), std::end(types), in_type);
    if (it == types.end()) {
        string type_str;
        for (int type_idx = 0; type_idx < types.size(); ++type_idx) {
            type_str += string(getEnumString(types[type_idx]));
            if (type_idx < types.size() - 1) {
                type_str += ", ";
            }
        }
        const string fmt_str = fmt::format("Array `{}` ({}) must be one of the following types: {}",
                                           in_str, getEnumString(in_type), type_str);
        throw AfError(function, file, line, fmt_str.c_str(), AF_ERR_TYPE);
    }
}
void
assert_type_eq(const af_dtype lhs_type, const af_dtype rhs_type,
               const char * lhs_str, const char * rhs_str,
               const char * file, const char * function, const int line)
{
    if (lhs_type != rhs_type) {
        const string fmt_str = fmt::format("Arrays `{}` ({}) and `{}` ({}) must be of the same type.",
                                           lhs_str, lhs_type, rhs_str, rhs_type);
        throw AfError(function, file, line, fmt_str.c_str(), AF_ERR_TYPE);
    }
}
void
type_error(const ArrayInfo &in_info, const char * in_str,
           const char * file, const char * function, const int line)
{
    const string fmt_str = fmt::format("Type of array `{}` ({}) is unsupported",
                                       in_str, getEnumString(in_info.getType()));
    throw AfError(function, file, line, fmt_str.c_str(), AF_ERR_TYPE);
}
void
unsupported_type(const af_dtype in, const char * in_str,
                 const char * file, const char * function, const int line)
{
    const string fmt_str = fmt::format("Type `{}` ({}) is unsupported", in_str, getEnumString(in));
    throw AfError(function, file, line, fmt_str.c_str(), AF_ERR_TYPE);
}

void
dim_eq(const ArrayInfo &lhs_info, const ArrayInfo &rhs_info,
       const char * lhs_str, const char * rhs_str,
       const char * file, const char * function, const int line)
{
    af::dim4 lhs_dims = lhs_info.dims();
    af::dim4 rhs_dims = rhs_info.dims();
    af_dtype lhs_type = lhs_info.getType();
    af_dtype rhs_type = rhs_info.getType();

    if (lhs_dims != rhs_dims) {
        string fmt_str = fmt::format("Arrays `{}` ({}) and `{}` ({}) must be of the same dimensions.", lhs_str, lhs_dims, rhs_str, rhs_dims);
        throw AfError(function, file, line, fmt_str.c_str(), AF_ERR_TYPE); // \TODO which af_err????
    }
}
void
dim_cmp(const int op,
        const int dim_idx,
        const int dim_val,
        const ArrayInfo &arr_info,
        const char * arr_str,
        const char * file, const char * function, const int line)
{
    af::dim4 arr_dims  = arr_info.dims();
    // std::cout << "arr_dims: " << arr_dims << endl;
    std::string op_str;
    switch(op) {
    case 0: {
        if (!(arr_dims[dim_idx] < dim_val))
            op_str = "less than"; break;
    }
    case 1: {
        if (!(arr_dims[dim_idx] == dim_val))
            op_str = "equal to"; break;
    };
    case 2: {
        if (!(arr_dims[dim_idx] > dim_val))
            op_str = "greather than"; break;
    }
    default: assert(0);
    }
    if (!op_str.empty()) {
        std::string fmt_str;
        fmt_str = fmt::format("The size of array `{}` ({}) must be {} {} on dimension {}", arr_str, arr_dims, op_str, dim_val, dim_idx);
        throw AfError(function, file, line, fmt_str.c_str(), AF_ERR_TYPE); // \TODO which af_err????
    }
}

void
print_error(const string &msg)
{
    std::string perr = getEnvVar("AF_PRINT_ERRORS");
    if(!perr.empty()) {
        if(perr != "0")
            fprintf(stderr, "%s\n", msg.c_str());
    }
    get_global_error_string() = msg;
}

af_err processException()
{
    stringstream    ss;
    af_err          err= AF_ERR_INTERNAL;

    try {
        throw;
    } catch (const DimensionError &ex) {
        ss << "In function " << ex.getFunctionName() << "\n"
           << "In file " << ex.getFileName() << ":" << ex.getLine() << "\n"
           << "Invalid dimension for argument " << ex.getArgIndex() << "\n"
           << "Expected: " << ex.getExpectedCondition() << "\n";

        print_error(ss.str());
        err = AF_ERR_SIZE;
    } catch (const ArgumentError &ex) {
        ss << "In function " << ex.getFunctionName() << "\n"
           << "In file " << ex.getFileName() << ":" << ex.getLine() << "\n"
           << "Invalid argument at index " << ex.getArgIndex() << "\n"
           << "Expected: " << ex.getExpectedCondition() << "\n";

        print_error(ss.str());
        err = AF_ERR_ARG;
    } catch (const SupportError &ex) {
        ss << ex.getFunctionName()
           << " not supported for " << ex.getBackendName()
           << " backend\n";

        print_error(ss.str());
        err = AF_ERR_NOT_SUPPORTED;
    } catch (const TypeError &ex) {
        ss << "In function " << ex.getFunctionName() << "\n"
           << "In file " << ex.getFileName() << ":" << ex.getLine() << "\n"
           << "Invalid type for argument " << ex.getArgIndex() << "\n";

        print_error(ss.str());
        err = AF_ERR_TYPE;
    } catch (const AfError &ex) {
        ss << "In function " << ex.getFunctionName() << "\n"
           << "In file " << ex.getFileName() << ":" << ex.getLine() << "\n"
           << ex.what() << "\n";

        print_error(ss.str());
        err = ex.getError();
#if defined(WITH_GRAPHICS) && !defined(AF_UNIFIED)
    } catch (const forge::Error &ex) {
        ss << ex << "\n";
        print_error(ss.str());
        err = AF_ERR_INTERNAL;
#endif
#ifdef AF_OPENCL
    } catch(const cl::Error &ex) {
      char opencl_err_msg[1024];
      snprintf(opencl_err_msg, sizeof(opencl_err_msg),
               "OpenCL Error (%d): %s when calling %s", ex.err(),
               getErrorMessage(ex.err()).c_str(), ex.what());
      print_error(opencl_err_msg);
      if (ex.err() == CL_MEM_OBJECT_ALLOCATION_FAILURE) {
        err = AF_ERR_NO_MEM;
      } else {
        err = AF_ERR_INTERNAL;
      }
#endif
    } catch (...) {
        print_error(ss.str());
        err = AF_ERR_UNKNOWN;
    }

    return err;
}

std::string& get_global_error_string()
{
    thread_local std::string *global_error_string = new std::string("");
    return *global_error_string;
}

const char *af_err_to_string(const af_err err)
{
    switch (err) {
    case AF_SUCCESS:                return "Success";
    case AF_ERR_NO_MEM:             return "Device out of memory";
    case AF_ERR_DRIVER:             return "Driver not available or incompatible";
    case AF_ERR_RUNTIME:            return "Runtime error ";
    case AF_ERR_INVALID_ARRAY:      return "Invalid array";
    case AF_ERR_ARG:                return "Invalid input argument";
    case AF_ERR_SIZE:               return "Invalid input size";
    case AF_ERR_TYPE:               return "Function does not support this data type";
    case AF_ERR_DIFF_TYPE:          return "Input types are not the same";
    case AF_ERR_BATCH:              return "Invalid batch configuration";
    case AF_ERR_NOT_SUPPORTED:      return "Function not supported";
    case AF_ERR_NOT_CONFIGURED:     return "Function not configured to build";
    case AF_ERR_NONFREE:            return "Function unavailable. "
                                           "ArrayFire compiled without Non-Free algorithms support";
    case AF_ERR_NO_DBL:             return "Double precision not supported for this device";
    case AF_ERR_NO_GFX:             return "Graphics functionality unavailable. "
                                           "ArrayFire compiled without Graphics support";
    case AF_ERR_LOAD_LIB:           return "Failed to load dynamic library. ";
    case AF_ERR_LOAD_SYM:           return "Failed to load symbol";
    case AF_ERR_ARR_BKND_MISMATCH:  return "There was a mismatch between an array and the current backend";
    case AF_ERR_INTERNAL:           return "Internal error";
    case AF_ERR_UNKNOWN:
    default:                        return "Unknown error";
    }
}
