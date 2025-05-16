#pragma once

#include <cuda.h>

#include <iostream>

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define C10_CUDA_CHECK(EXPR)                                                                        \
  do {                                                                                              \
    const cudaError_t __err = EXPR;                                                                 \
    if (__err != cudaSuccess) {                                                                     \
      std::cerr << "CUDA Error: " << cudaGetErrorString(__err) << " (" << __err << ") " << __FILE__ \
                << ": line " << __LINE__ << " at function " << STR(func) << std::endl;              \
      return;                                                                                       \
    }                                                                                               \
  } while (0)

// This should be used directly after every kernel launch to ensure
// the launch happened correctly and provide an early, close-to-source
// diagnostic if it didn't.
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
