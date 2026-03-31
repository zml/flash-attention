#pragma once

#include "capi.h"

#include <cassert>
#include <span>
#include <algorithm>
#include <sstream>

#define CAPI_CHECK(cond, ...)                                                                 \
    do {                                                                                      \
        if (!(cond)) {                                                                        \
            fprintf(stderr, "Check failed (%s:%d): %s\n", __FILE__, __LINE__, ##__VA_ARGS__); \
            exit(1);                                                                          \
        }                                                                                     \
    } while(0)

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define CUDA_CALL(func, ...)                                                                \
  {                                                                                         \
    cudaError_t e = (func);                                                                 \
    if (e != cudaSuccess) {                                                                 \
      std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " (" << e << ") " << __FILE__ \
                << ": line " << __LINE__ << " at function " << STR(func) << std::endl;      \
      exit(1);                                                                              \
    }                                                                                       \
  }

#define CAPI_CHECK_SHAPE(x, ...) CAPI_CHECK(equalShape(x, std::span<const int64_t>({__VA_ARGS__})), #x " must have shape (" #__VA_ARGS__ ")")

inline bool equalShape(const FlashattnTensor* tensor, std::span<const int64_t> dims) {
    if (tensor->rank != dims.size()) return false;
    std::span<const int64_t> tensor_dims{tensor->dims, static_cast<size_t>(tensor->rank)};
    return std::equal(tensor_dims.begin(), tensor_dims.end(), dims.begin());
}

inline int64_t getStride(const FlashattnTensor *tensor, int32_t axis) {
    if (axis < 0) {
        axis += tensor->rank;
    }
    CAPI_CHECK(axis >= 0 && axis < tensor->rank, "Expected axis to be within bounds");
    return tensor->strides[axis];
}

inline int64_t getDim(const FlashattnTensor *tensor, int32_t axis) {
    if (axis < 0) {
        axis += tensor->rank;
    }
    CAPI_CHECK(axis >= 0 && axis < tensor->rank, "Expected axis to be within bounds");
    return tensor->dims[axis];
}

inline int64_t numElements(const FlashattnTensor *tensor) {
    int64_t size = 1;
    for (int i = 0;i < tensor->rank;i++) {
        size *= tensor->dims[i];
    }

    return size;
}

inline int64_t dtypeByteSize(DataType dtype) {
    switch (dtype) {
        case CAPI_BFLOAT16:
        case CAPI_FLOAT16:
            return 2;
        break;
        case CAPI_INT32:
        case CAPI_FLOAT:
            return 4;
        break;
        case CAPI_INT8:
            return 1;
        break;
        default:
        CAPI_CHECK(false, "Unsupported datatype");
        break;
    }
}

inline int64_t byteSize(const FlashattnTensor *tensor) {
    return numElements(tensor) * dtypeByteSize(tensor->dtype);
}

inline void printTensor(FlashattnTensor tensor, const char* name) {
    std::ostringstream stream;
    stream << name << ":\n";
    stream << "    dtype: ";
    switch (tensor.dtype) {
        case CAPI_F8E4M3FN:
            stream << "bf16";
            break;
        case CAPI_BFLOAT16:
            stream << "bf16";
            break;
        case CAPI_FLOAT16:
            stream << "f16";
            break;
        case CAPI_FLOAT:
            stream << "f32";
            break;
        case CAPI_INT32:
            stream << "i32";
            break;
        case CAPI_INT8:
            stream << "i8";
            break;
        default:
            stream << "<unk>";
            break;
    }
    stream << "\n";
    stream << "    rank: " << tensor.rank << "\n";
    stream << "    dims: ["; 
    for (int i = 0;i < tensor.rank;i++) {
        if (i != 0) stream << ", ";
        stream << tensor.dims[i];
    }
    stream << "]\n";
    stream << "    strides: ["; 
    for (int i = 0;i < tensor.rank;i++) {
        if (i != 0) stream << ", ";
        stream << tensor.strides[i];
    }
    stream << "]\n";
    stream << "    ptr: " << tensor.ptr << "\n";
    std::cout << stream.str();
}
