# NOTES

## Goal

Build a minimal, repeatable repro for FA3 forward-pass misbehavior seen from a custom Llama implementation, with FA2 as the known-good comparison point.

## Current Status

- Branch: `llvm-repro`
- Initial blockers found:
  - The checkout was missing `csrc/cutlass` and `csrc/composable_kernel` submodules.
  - `//:fa3_sm90_repro` did not build before submodule init.
  - After submodule init, the reduced repro target still failed to link because `capi-sm90-repro` could dispatch paged-KV FA3 kernels that `flashattn-sm90-repro` does not compile.
- Current fix in progress:
  - Repro-only Bazel targets now disable paged-KV dispatch to match the reduced kernel set.
  - `tests/fa3_sm90_repro.cc` now drives FA3 with varlen-shaped inputs and `cu_seqlens_*` by default, which matches the `FLASHATTENTION_VARLEN_ONLY` repro build.
  - Current result: the binary builds, but FA3 still fails at runtime with `CUDA error: unspecified launch failure` on `cudaDeviceSynchronize()`, even for tiny shapes.

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
- Confirmed the original harness bug:
  - The prior harness called FA3 with dense 4D tensors and no `cu_seqlens_*` despite `FLASHATTENTION_VARLEN_ONLY`.
  - That configuration is not representative for this repro target.
- Rebuilt after fixing the harness to use varlen-shaped tensors:
  - Default config still fails.
  - `--seqlen_q=1 --seqlen_k=1 --num_heads=4 --num_heads_k=4 --head_dim=128 --iters=1` still fails.
  - `--causal=0` still fails.
- Current runtime invocation:
  - Direct execution requires Bazel CUDA redist in `LD_LIBRARY_PATH`.
  - Using the Bazel-downloaded CUDA 13 runtime resolves the loader error, then the FA3 kernel crashes.

## Next Steps

1. Compare this C API FA3 failure against the higher-level FA3 torch op path to determine whether the bug is in `capi/capi_sm90.cc` or deeper in the kernel/runtime path.
2. Extend `tests/fa3_sm90_repro.cc` to compare FA2 and FA3 on the same inputs once the FA3 call surface is stable.
3. Add targeted switches that better match the custom Llama surface:
   - dense vs varlen
   - GQA ratios
   - `seqused_*`
   - split count
   - first failing config / seed reporting
4. Sweep configs until FA3 diverges while FA2 stays correct, or identify the exact wrapper misuse / missing parameter that causes the crash.

## Open Questions

- Does the custom Llama path use dense tensors, varlen tensors, or `seqused_*`?
- Is the observed randomness true nondeterminism across repeated runs, or deterministic-but-wrong outputs?
- Is the failure tied to GQA, causal masking, or a specific sequence-length regime?
