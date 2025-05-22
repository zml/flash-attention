#pragma once

#include "capi.h"

#include <cassert>

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

inline int64_t getStride(FlashattnTensor tensor, int32_t axis) {
    if (axis < 0) {
        axis += tensor.rank;
    }
    CAPI_CHECK(axis >= 0 && axis < tensor.rank, "Expected axis to be within bounds");
    return tensor.strides[axis];
}

inline int64_t getDim(FlashattnTensor tensor, int32_t axis) {
    if (axis < 0) {
        axis += tensor.rank;
    }
    CAPI_CHECK(axis >= 0 && axis < tensor.rank, "Expected axis to be within bounds");
    return tensor.dims[axis];
}

inline int64_t numElements(FlashattnTensor tensor) {
    int64_t size = 1;
    for (int i = 0;i < tensor.rank;i++) {
        size *= tensor.dims[i];
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

inline int64_t byteSize(FlashattnTensor tensor) {
    return numElements(tensor) * dtypeByteSize(tensor.dtype);
}
