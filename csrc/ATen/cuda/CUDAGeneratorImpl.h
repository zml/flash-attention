#pragma once

namespace at {

struct PhiloxCudaState {
  PhiloxCudaState() = default;
  // Called if graph capture is not underway
  PhiloxCudaState(uint64_t seed,
                  uint64_t offset) {
    seed_.val = seed;
    offset_.val = offset;
  }
  // Called if graph capture is underway
  PhiloxCudaState(int64_t* seed,
                  int64_t* offset_extragraph,
                  uint32_t offset_intragraph) {
    seed_.ptr = seed;
    offset_.ptr = offset_extragraph;
    offset_intragraph_ = offset_intragraph;
    captured_ = true;
  }

  // Public members, directly accessible by at::cuda::philox::unpack.
  // If we made them private with getters/setters, the getters/setters
  // would have to be __device__, and we can't declare __device__ in ATen.
  union Payload {
    uint64_t val;
    int64_t* ptr;
  };

  Payload seed_{};
  Payload offset_{};
  uint32_t offset_intragraph_ = 0;
  bool captured_ = false;
};

} // namespace at
