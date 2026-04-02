# NOTES

## Goal

Build a minimal, repeatable repro for FA3 forward-pass misbehavior seen from a custom Llama implementation, with FA2 as the known-good comparison point.

## Current Status

- Branch: `llvm-repro`
- Initial blockers found:
  - The checkout was missing `csrc/cutlass` and `csrc/composable_kernel` submodules.
  - `//:fa3_sm90_repro` did not build before submodule init.
  - After submodule init, the reduced repro target still failed to link because `capi-sm90-repro` could dispatch paged-KV FA3 kernels that `flashattn-sm90-repro` does not compile.
- Current repro surface:
  - `tests/fa3_sm90_repro.cc` now supports both the reduced varlen-only path and a paged-KV path with `page_table` plus optional `seqused_k`.
  - Added a full clang SM90 target in `BUILD.bazel`: `//:fa3_sm90_full_repro`.
  - `.bazelrc` remote default is now `--jobs=1000`.
- Current result:
  - `//:fa3_sm90_full_repro` builds successfully with `bazel build --config remote --jobs=1000`.
  - The full clang-built FA3 path still fails at runtime with `CUDA error: unspecified launch failure` on `cudaDeviceSynchronize()` before any output comparison.

## What Has Been Tested

- Located the prior repro target in `BUILD.bazel`:
  - `//:fa3_sm90_repro`
  - `tests/fa3_sm90_repro.cc`
- Confirmed hardware and toolchain:
  - GPU: H100 80GB
  - Bazel available through Bazelisk
- Read `capi/capi_sm90.cc`:
  - FA3 wrapper always requires `scheduler_metadata` when the scheduler path needs it.
  - Repro build is using `FLASHATTENTION_VARLEN_ONLY`.
  - Runtime dispatch can instantiate paged-KV and pack-GQA branches unless explicitly compiled out.
  - Paged-KV mode in the SM90 C API requires rank-4 K/V tensors plus `page_table`, and rejects `cu_seqlens_k`.
- Confirmed the original harness bug:
  - The prior harness called FA3 with dense 4D tensors and no `cu_seqlens_*` despite `FLASHATTENTION_VARLEN_ONLY`.
  - That configuration is not representative for this repro target.
- Rebuilt after fixing the harness to use varlen-shaped tensors:
  - Default config still fails.
  - `--seqlen_q=1 --seqlen_k=1 --num_heads=4 --num_heads_k=4 --head_dim=128 --iters=1` still fails.
  - `--causal=0` still fails.
- Added a paged-KV harness mode in `tests/fa3_sm90_repro.cc`:
  - `--paged_kv=0|1`
  - `--page_size=N`
  - `--seqused_k=N`
  - In paged mode, the harness generates logical K/V, packs it into page storage, and passes an identity `page_table`.
- Built and ran the full clang SM90 target:
  - Build command:
    - `bazel build --config remote --jobs=1000 //:fa3_sm90_full_repro`
  - BuildBuddy invocation:
    - `e1acc423-56da-40bb-ac94-8d297e1bd3f9`
  - Small full-target varlen+paged config:
    - `--iters=2 --batch=1 --seqlen_q=8 --seqlen_k=8 --num_heads=4 --num_heads_k=4 --head_dim=128 --causal=1 --varlen=1 --paged_kv=1 --page_size=16`
    - Result: `CUDA error: unspecified launch failure` at `tests/fa3_sm90_repro.cc:477`
  - Same full-target geometry with paged KV disabled:
    - `--iters=2 --batch=1 --seqlen_q=8 --seqlen_k=8 --num_heads=4 --num_heads_k=4 --head_dim=128 --causal=1 --varlen=1 --paged_kv=0`
    - Result: same launch failure at the same line
  - Custom-llama-like head geometry:
    - `--iters=1 --batch=1 --seqlen_q=64 --seqlen_k=64 --num_heads=32 --num_heads_k=8 --head_dim=128 --causal=1 --varlen=1 --paged_kv=1 --page_size=1024`
    - Result: same launch failure at the same line
- Current runtime invocation:
  - Direct execution requires Bazel CUDA redist in `LD_LIBRARY_PATH`.
  - Using the Bazel-downloaded CUDA 13 runtime resolves the loader error, then the FA3 kernel crashes.
  - CUDA redist path used so far:
    - `/root/.cache/bazel/_bazel_root/cache/repos/v1/contents/b84070b7338ac3d07a942ea84b3d3ea67db8e867f439a345cf9acd7e9b9eeef2/079b9eaf-27d6-4139-9001-7b31e6d79fed/lib`

## Custom Llama IR Clues

- User-provided typed-FFI custom call:
  - output `bf16[2048,32,128]`
  - Q `bf16[2048,32,128]`
  - K/V `bf16[2048,8,128]`
  - extra `s32[2]` operands plus scratch buffers
  - `is_causal = true`
  - `max_seqlen_q = 2048`
  - `max_seqlen_k = 2048`
- Best current interpretation:
  - batch is effectively 1
  - GQA ratio is `32 / 8 = 4`
  - at least one `s32[2]` operand is consistent with `cu_seqlens_q = [0, 2048]`
  - another `s32[2]` operand is plausibly a `page_table` with 2 entries if page size is 1024
- This is not yet proven from local C++ code because the typed-FFI lowering is not in this repo.

## Next Steps

1. Keep working only from C++/Bazel and the pasted IR surface; avoid Python.
2. Add a way to skip the CPU reference for very large configs so the exact `2048 x 32 x 128` / `2048 x 8 x 128` geometry can be executed directly.
3. Probe whether the crash is tied to a specific option boundary:
   - `varlen=0|1`
   - `paged_kv=0|1`
   - `page_size`
   - `seqused_k`
   - `causal=0|1`
   - `num_heads=32`, `num_heads_k=8`, `head_dim=128`
4. If a non-crashing clang config is found, compare against CPU reference immediately and record the first wrong-output seed/config.
5. If every full-target clang config still crashes, narrow the failure to a smaller C API surface change or kernel family and keep checkpointing often.

## Open Questions

- Which typed-FFI operands in the custom call correspond to `page_table`, `scheduler_metadata`, and scratch buffers?
- Does the custom Llama path pass paged KV with physically rank-4 storage that is bitcast to rank-3 in HLO, or is there a separate lowering layer reshaping it?
- Is there any clang-built FA3 config on this branch that survives the first sync, or is the current failure unconditional across the SM90 forward path?
