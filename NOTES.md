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
  - `--skip_ref=0|1`
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
  - Minimal dense/non-paged/non-causal config:
    - `--batch=1 --seqlen_q=1 --seqlen_k=1 --num_heads=4 --num_heads_k=4 --head_dim=128 --causal=0 --varlen=0 --paged_kv=0 --skip_ref=1`
    - Result: same launch failure at `tests/fa3_sm90_repro.cc:482`
  - Minimal varlen/non-paged/non-causal config:
    - `--batch=1 --seqlen_q=1 --seqlen_k=1 --num_heads=4 --num_heads_k=4 --head_dim=128 --causal=0 --varlen=1 --paged_kv=0 --skip_ref=1`
    - Result: same launch failure at `tests/fa3_sm90_repro.cc:482`
  - Minimal varlen/paged/non-causal config:
    - `--batch=1 --seqlen_q=1 --seqlen_k=1 --num_heads=4 --num_heads_k=4 --head_dim=128 --causal=0 --varlen=1 --paged_kv=1 --page_size=16 --skip_ref=1`
    - Result: same launch failure at `tests/fa3_sm90_repro.cc:482`
  - Custom-llama-like head geometry:
    - `--iters=1 --batch=1 --seqlen_q=64 --seqlen_k=64 --num_heads=32 --num_heads_k=8 --head_dim=128 --causal=1 --varlen=1 --paged_kv=1 --page_size=1024`
    - Result: same launch failure at the same line
  - Exact IR-sized geometry without CPU reference:
    - `--iters=1 --batch=1 --seqlen_q=2048 --seqlen_k=2048 --num_heads=32 --num_heads_k=8 --head_dim=128 --causal=1 --varlen=1 --paged_kv=1 --page_size=1024 --skip_ref=1`
    - Result: same launch failure at `tests/fa3_sm90_repro.cc:482`
- Current runtime invocation:
  - Direct execution requires Bazel CUDA redist in `LD_LIBRARY_PATH`.
  - Using the Bazel-downloaded CUDA 13 runtime resolves the loader error, then the FA3 kernel crashes.
  - CUDA redist path used so far:
    - `/root/.cache/bazel/_bazel_root/cache/repos/v1/contents/b84070b7338ac3d07a942ea84b3d3ea67db8e867f439a345cf9acd7e9b9eeef2/079b9eaf-27d6-4139-9001-7b31e6d79fed/lib`
  - `bazel aquery //:fa3_sm90_full_repro` confirms the target is built through the LLVM CUDA toolchain and uses `clang++` from the Bazel-managed LLVM toolchain inputs.

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
4. The exact IR-sized paged-KV config is now directly runnable, and it still crashes.
5. Shrinking below that surface still crashes, including the smallest dense/non-paged/non-causal case tested so far.
6. Next useful control is FA2 through the same C API harness, to prove the harness path is sound on the same machine while FA3 clang remains broken.
7. If a non-crashing clang FA3 config is found, compare against CPU reference immediately and record the first wrong-output seed/config.

## Open Questions

- Which typed-FFI operands in the custom call correspond to `page_table`, `scheduler_metadata`, and scratch buffers?
- Does the custom Llama path pass paged KV with physically rank-4 storage that is bitcast to rank-3 in HLO, or is there a separate lowering layer reshaping it?
- Is there any clang-built FA3 config on this branch that survives the first sync, or is the current failure unconditional across the SM90 forward path?

## 2026-04-02 Dynamic-Split Findings

- Added temporary FA3 dispatch logging in `capi/capi_sm90.cc`, enabled with `FA3_REPRO_DEBUG=1`.
- This exposed a new non-crashing wrong-output repro on the clang-built full SM90 FA3 binary:
  - Build:
    - `bazel build --config remote --jobs=1000 //:fa3_sm90_full_repro`
  - BuildBuddy invocation:
    - `d8716b41-32ab-4fdf-afdf-811afe92e54b`
  - Small paged-KV config with heuristic splits:
    - `FA3_REPRO_DEBUG=1 LD_LIBRARY_PATH=... ./bazel-bin/fa3_sm90_full_repro --batch=1 --seqlen_q=64 --seqlen_k=64 --num_heads=32 --num_heads_k=8 --head_dim=128 --causal=1 --varlen=1 --paged_kv=1 --page_size=16 --num_splits=0`
    - Result: no crash, but output is deterministically all zeros and fails the CPU reference.
    - Compare summary:
      - `max_abs=0.250000`
      - `max_rel=1.000000`
      - `mean_abs=0.025934`
      - sample: all zeros
  - Larger exact llama-like paged-KV config with heuristic splits:
    - `FA3_REPRO_DEBUG=1 LD_LIBRARY_PATH=... ./bazel-bin/fa3_sm90_full_repro --batch=1 --seqlen_q=2048 --seqlen_k=2048 --num_heads=32 --num_heads_k=8 --head_dim=128 --causal=1 --varlen=1 --paged_kv=1 --page_size=1024 --num_splits=0 --skip_ref=1`
    - Result: no crash, but sample output is also all zeros.
- The same geometry with explicit `--num_splits=1` still crashes:
  - `FA3_REPRO_DEBUG=1 ... --seqlen_q=64 --seqlen_k=64 --paged_kv=1 --page_size=16 --num_splits=1 --skip_ref=1`
  - Result: `CUDA error: unspecified launch failure`
- The debug logs show:
  - `--num_splits=1`:
    - `paged_kv=1 pagedkv_tma=0 num_splits=1 use_dynamic_split=0 scheduler_needs_semaphore=1 pack_gqa=1`
  - `--num_splits=0` on the same small case:
    - `paged_kv=1 pagedkv_tma=0 num_splits=1 use_dynamic_split=1 scheduler_needs_semaphore=1 pack_gqa=1`
  - `--num_splits=0` on the exact `2048 x 32 x 128` / `2048 x 8 x 128` case:
    - `paged_kv=1 pagedkv_tma=1 num_splits=1 use_dynamic_split=1 scheduler_needs_semaphore=1 pack_gqa=1`
- Current interpretation:
  - The failing behavior is strongly tied to the paged-KV varlen scheduler mode selection.
  - Static scheduler path (`use_dynamic_split=0`) crashes.
  - Dynamic-split scheduler path (`use_dynamic_split=1`) survives but produces all-zero outputs.
  - This is much closer to the user's real llama symptom than the earlier "always crashes" conclusion.
- Additional note from the debug print:
  - `capi/capi_sm90.cc` reports `total_k=batch_size * getDim(k, 1)` even for paged KV, so for `seqlen_k=64,page_size=16` it prints `total_k=16`, and for `seqlen_k=2048,page_size=1024` it prints `total_k=1024`.
  - This may or may not be causal for forward correctness, but it is inconsistent with the logical KV length and should be audited.
- Follow-up audit:
  - Patched `capi/capi_sm90.cc` so paged-KV reports logical `total_k = batch_size_k * seqlen_k`.
  - Rebuilt `//:fa3_sm90_full_repro` with invocation:
    - `0b94dd11-9902-47fd-a24e-eb09320df3e9`
  - Reran the same two FA3 dynamic-split repros.
  - Result:
    - debug print now shows correct `total_k` (`64` and `2048`)
    - behavior is unchanged
    - small paged-KV case still returns deterministic all-zero outputs and fails reference
    - exact llama-like paged-KV case still returns all-zero sampled outputs
  - Conclusion:
    - the incorrect paged-KV `total_k` accounting was a real wrapper bug, but it is not sufficient to explain the observed clang FA3 wrong-output path.
- FA2 control status:
  - Tried building `//:fa2_llama_repro`, but the reduced FA2 target currently fails to link because it depends on the full `:capi` dispatcher, which references many FA2 kernels not present in the reduced FA2 library.
  - This is a separate build-system issue in the control target, not yet a runtime result.

## 2026-04-02 Scheduler / PDL Narrowing

- Added scheduler buffer dumps in `tests/fa3_sm90_repro.cc` under `FA3_REPRO_DEBUG=1`.
- Added a temporary escape hatch in `capi/capi_sm90.cc`:
  - `FA3_REPRO_SKIP_SCHED_PREP=1`
  - This sets `params.skip_scheduler_metadata_computation = true`
  - The harness then preloads `scheduler_metadata` with `[0, 1, ...]` so the main kernel sees:
    - semaphore = `0`
    - `num_splits_dynamic = 1`
- Small paged-KV dynamic-split repro with scheduler prep enabled:
  - `FA3_REPRO_DEBUG=1 LD_LIBRARY_PATH=... ./bazel-bin/fa3_sm90_full_repro --batch=1 --seqlen_q=64 --seqlen_k=64 --num_heads=32 --num_heads_k=8 --head_dim=128 --causal=1 --varlen=1 --paged_kv=1 --page_size=16 --num_splits=0 --dump_count=8`
  - Observed debug line:
    - `paged_kv=1 pagedkv_tma=0 num_splits=1 use_dynamic_split=1 scheduler_needs_semaphore=1 pack_gqa=1 skip_sched_meta=0 total_k=64`
  - Scheduler dump after each iteration:
    - `sched 0 1`
    - `sched 0 1`
    - `sched 0 1`
  - Result:
    - no crash
    - output remains deterministically all zeros
    - semaphore never advances beyond `0`
- Main-kernel PDL experiment:
  - Patched `hopper/flash_fwd_launch_template.h` so the main `cutlass::kernel_launch<AttnKernel>(...)` uses `launch_with_pdl=false`
  - Result:
    - the same small dynamic-split paged-KV repro now crashes with `CUDA error: unspecified launch failure`
    - the exact llama-like paged-KV case (`2048 x 32 x 128`, `2048 x 8 x 128`, `page_size=1024`, `num_splits=0`) also crashes
  - Interpretation:
    - PDL changes the symptom:
      - main-kernel PDL on => wrong all-zero outputs
      - main-kernel PDL off => launch failure
- Scheduler-prep bypass experiment with main-kernel PDL restored:
  - Restored the original main-kernel PDL expression.
  - Kept prep-kernel PDL disabled:
    - `prepare_varlen_num_blocks(..., false /*enable_pdl*/)`
  - Built with invocation:
    - `ab2ac4cb-0093-4d63-b537-071f34f11b0f`
  - Ran:
    - `FA3_REPRO_DEBUG=1 FA3_REPRO_SKIP_SCHED_PREP=1 LD_LIBRARY_PATH=... ./bazel-bin/fa3_sm90_full_repro --batch=1 --seqlen_q=64 --seqlen_k=64 --num_heads=32 --num_heads_k=8 --head_dim=128 --causal=1 --varlen=1 --paged_kv=1 --page_size=16 --num_splits=0 --dump_count=8`
  - Result:
    - `CUDA error: unspecified launch failure`
  - Interpretation:
    - the failure is not only in `prepare_varlen_num_blocks`
    - the crash survives when scheduler prep is bypassed and the scheduler buffer is host-seeded
- Current best narrowing:
  - The reproducible wrong-output surface is:
    - FA3
    - SM90
    - clang-built binary
    - varlen + paged-KV
    - dynamic-split scheduler path
  - The remaining likely root-cause area is the main persistent scheduler kernel path, especially its interaction with PDL under clang-generated Hopper code.

## Immediate Next Step

1. Add launch-level debug prints in `hopper/flash_fwd_launch_template.h`:
   - `num_blocks_m`
   - grid and block dims
   - shared memory size
   - `tile_count_semaphore` pointer
   - `num_splits_dynamic_ptr` pointer
   - whether the scheduler class is persistent
2. Rebuild and rerun the small paged-KV dynamic-split repro with `FA3_REPRO_DEBUG=1`.
3. Use that data to decide whether the main kernel is being launched with inconsistent scheduler metadata or whether the problem is already deeper in clang-generated device code.

## 2026-04-02 Post-Launch Narrowing

- Added launch-level debug prints in `hopper/flash_fwd_launch_template.h` under `FA3_REPRO_DEBUG=1`:
  - `num_blocks_m`
  - grid dims
  - block dims
  - dynamic shared memory size
  - persistent vs single-tile scheduler choice
  - `tile_count_semaphore`
  - `num_splits_dynamic_ptr`
  - `cu_seqlens_q`
- Rebuilt `//:fa3_sm90_full_repro` with invocation:
  - `2493cd24-39a5-410b-bd73-cd41a902bdb7`
- Small paged-KV dynamic-split repro with launch debug:
  - `FA3_REPRO_DEBUG=1 LD_LIBRARY_PATH=... ./bazel-bin/fa3_sm90_full_repro --batch=1 --seqlen_q=64 --seqlen_k=64 --num_heads=32 --num_heads_k=8 --head_dim=128 --causal=1 --varlen=1 --paged_kv=1 --page_size=16 --num_splits=0 --dump_count=8`
  - Observed launch:
    - `persistent=1`
    - `num_blocks_m=2`
    - `grid=(132,1,1)`
    - `block=(384,1,1)`
    - valid non-null device pointers for `tile_sem`, `num_splits_dyn`, and `cu_q`
  - Result:
    - output still all zeros
    - scheduler metadata still remains `sched 0 1`
- Small paged-KV dynamic-split repro with prep bypass:
  - `FA3_REPRO_DEBUG=1 FA3_REPRO_SKIP_SCHED_PREP=1 ...`
  - Launch geometry is the same as above
  - Result:
    - immediate `CUDA error: unspecified launch failure`
- Small paged-KV explicit split repro:
  - `FA3_REPRO_DEBUG=1 ... --num_splits=1 --skip_ref=1`
  - Launch geometry is the same except `num_splits_dyn=(nil)`
  - Result:
    - immediate `CUDA error: unspecified launch failure`
- Interpretation after launch-level instrumentation:
  - the wrapper is passing coherent-looking scheduler pointers and launch geometry
  - the bad behavior is not explained by obviously bad scheduler metadata at launch time

## 2026-04-02 Persistent Scheduler Elimination

- Forced varlen FA3 onto `SingleTileScheduler` in `hopper/flash_fwd_launch_template.h` to remove `VarlenDynamicPersistentTileScheduler` from the equation.
- Rebuilt with invocation:
  - `2ee10916-bd5e-42bd-836d-3d47512ec7fe`
- Small paged-KV dynamic-split repro after forcing `SingleTileScheduler`:
  - launch changes to:
    - `persistent=0`
    - `grid=(2,8,1)`
  - Result:
    - output is still deterministically all zeros
    - scheduler metadata still shows `sched 0 1`
- Small non-paged varlen repro with the same head geometry:
  - `--batch=1 --seqlen_q=64 --seqlen_k=64 --num_heads=32 --num_heads_k=8 --head_dim=128 --causal=1 --varlen=1 --paged_kv=0 --num_splits=0`
  - Result:
    - still deterministically all zeros
- Small non-paged varlen repro without GQA:
  - `--batch=1 --seqlen_q=64 --seqlen_k=64 --num_heads=32 --num_heads_k=32 --head_dim=128 --causal=1 --varlen=1 --paged_kv=0 --num_splits=0`
  - Result:
    - still deterministically all zeros
- Minimal wrong-output repro found:
  - `FA3_REPRO_DEBUG=1 LD_LIBRARY_PATH=... ./bazel-bin/fa3_sm90_full_repro --batch=1 --seqlen_q=64 --seqlen_k=64 --num_heads=1 --num_heads_k=1 --head_dim=64 --causal=1 --varlen=1 --paged_kv=0 --num_splits=0 --dump_count=8`
  - Launch:
    - `persistent=0`
    - `grid=(1,1,1)`
    - `block=(256,1,1)`
  - Result:
    - deterministic all-zero output
    - fails CPU reference
- Current best narrowing:
  - The wrong-output bug no longer requires:
    - paged KV
    - GQA / PackGQA
    - dynamic persistent scheduling
  - The remaining minimal bad surface is:
    - SM90 FA3
    - clang build
    - varlen forward path
    - batch=1
    - seqlen_q=64
    - seqlen_k=64
    - num_heads=1
    - num_heads_k=1
    - head_dim=64
  - This points away from scheduler metadata and toward the varlen kernel specialization / mainloop path itself under clang.
- Additional reduction:
  - The same minimal repro still returns all zeros with `--causal=0`.
  - So causality is not required.
- Critical split on the minimal repro:
  - `--num_splits=0`:
    - `use_dynamic_split=1`
    - `num_splits_dyn` is non-null
    - launch stays `persistent=0`, `grid=(1,1,1)`, `block=(256,1,1)`
    - result: deterministic all-zero output
  - `--num_splits=1`:
    - `use_dynamic_split=0`
    - `num_splits_dyn=(nil)`
    - launch still stays `persistent=0`, `grid=(1,1,1)`, `block=(256,1,1)`
    - result: `CUDA error: unspecified launch failure`
- Current interpretation:
  - On the reduced `SingleTileScheduler` repro, the zeros-vs-crash difference survives even after removing paged KV, GQA, and the persistent scheduler.
  - The remaining material runtime difference is the main-kernel PDL launch condition:
    - `launch_with_pdl=true` when varlen + `num_splits_dynamic_ptr` is present
    - `launch_with_pdl=false` when it is absent
  - That makes the strongest current root-cause hypothesis:
    - clang-generated SM90 FA3 varlen kernel behavior around the main-kernel PDL path, not scheduler metadata layout.

## 2026-04-02 Direct PDL Confirmation

- Patched the reduced `SingleTileScheduler` build so the main FA3 kernel always launches with PDL when `Varlen=true`.
- Rebuilt with invocation:
  - `bbb2f49a-b0f9-4666-999e-8a9d8e33c11d`
- Reran the minimal explicit-split case:
  - `FA3_REPRO_DEBUG=1 LD_LIBRARY_PATH=... ./bazel-bin/fa3_sm90_full_repro --batch=1 --seqlen_q=64 --seqlen_k=64 --num_heads=1 --num_heads_k=1 --head_dim=64 --causal=0 --varlen=1 --paged_kv=0 --num_splits=1 --skip_ref=1 --dump_count=8`
  - Before this patch:
    - same config crashed with `CUDA error: unspecified launch failure`
  - After forcing main-kernel PDL on:
    - no crash
    - output becomes deterministic all zeros
    - scheduler dump is `sched 0 0`
    - sample output remains all zeros
- Conclusion:
  - The crash-vs-zeros split is directly controlled by the main-kernel PDL launch path on the reduced varlen repro.
  - Since this repro already removed paged KV, GQA, and the persistent scheduler, the strongest current root-cause statement is:
    - clang-built SM90 FA3 varlen kernels are broken around the main-kernel PDL path
    - `launch_with_pdl=false` manifests as launch failure
    - `launch_with_pdl=true` manifests as silent all-zero output

## 2026-04-02 GDC Wait Isolation

- Restored the original host-side PDL condition in `hopper/flash_fwd_launch_template.h`.
- Disabled only the unconditional `cutlass::arch::wait_on_dependent_grids()` in `hopper/flash_fwd_kernel_sm90.h` for the reduced repro build.
- Rebuilt with invocation:
  - `0757ba25-3b49-40a5-b074-cfefce382240`
- Minimal explicit-split case after removing `griddepcontrol.wait`:
  - `FA3_REPRO_DEBUG=1 LD_LIBRARY_PATH=... ./bazel-bin/fa3_sm90_full_repro --batch=1 --seqlen_q=64 --seqlen_k=64 --num_heads=1 --num_heads_k=1 --head_dim=64 --causal=0 --varlen=1 --paged_kv=0 --num_splits=1 --skip_ref=1 --dump_count=8`
  - Result:
    - still `CUDA error: unspecified launch failure`
- Minimal heuristic-split case after removing `griddepcontrol.wait`:
  - same config but `--num_splits=0`
  - Result:
    - still deterministic all-zero output
    - scheduler dump still `sched 0 1`
- Conclusion:
  - `griddepcontrol.wait` alone is not sufficient to explain either symptom.
  - The remaining highest-value compiler/runtime boundary is the broader SM90 GDC/PDL enablement path itself, not just the single wait instruction.

## 2026-04-02 Return To Real Path

- Removed the reduction-only hacks and restored the real SM90 FA3 path:
  - restored normal `UsePersistentScheduler` selection in `hopper/flash_fwd_launch_template.h`
  - restored `cutlass::arch::wait_on_dependent_grids()` in `hopper/flash_fwd_kernel_sm90.h`
  - restored `CUTLASS_ENABLE_GDC_FOR_SM90` in `BUILD.bazel`
  - removed the temporary `FA3_REPRO_SKIP_SCHED_PREP` bypass path from `capi/capi_sm90.cc` and `tests/fa3_sm90_repro.cc`
- Rebuilt the real path with invocation:
  - `64f1d6c1-9001-4dcd-9625-2d5ed7baae1a`
- Exact llama-like non-crashing FA3 surface, non-paged:
  - `FA3_REPRO_DEBUG=1 LD_LIBRARY_PATH=... ./bazel-bin/fa3_sm90_full_repro --batch=1 --seqlen_q=2048 --seqlen_k=2048 --num_heads=32 --num_heads_k=8 --head_dim=128 --causal=1 --varlen=1 --paged_kv=0 --num_splits=0 --skip_ref=1 --dump_count=16`
  - Result:
    - no crash
    - launch uses the real path again:
      - `persistent=1`
      - `pack_gqa=1`
      - `use_dynamic_split=1`
    - sampled output is all zeros
    - scheduler metadata remains `sched 0 1`
- Exact llama-like non-crashing FA3 surface, paged KV:
  - `FA3_REPRO_DEBUG=1 LD_LIBRARY_PATH=... ./bazel-bin/fa3_sm90_full_repro --batch=1 --seqlen_q=2048 --seqlen_k=2048 --num_heads=32 --num_heads_k=8 --head_dim=128 --causal=1 --varlen=1 --paged_kv=1 --page_size=1024 --num_splits=0 --skip_ref=1 --dump_count=16`
  - Result:
    - no crash
    - sampled output is all zeros
    - scheduler metadata remains `sched 0 1`
- Smaller real-path reference-enabled repros:
  - non-paged:
    - `--batch=1 --seqlen_q=64 --seqlen_k=64 --num_heads=32 --num_heads_k=8 --head_dim=128 --causal=1 --varlen=1 --paged_kv=0 --num_splits=0`
  - paged:
    - `--batch=1 --seqlen_q=64 --seqlen_k=64 --num_heads=32 --num_heads_k=8 --head_dim=128 --causal=1 --varlen=1 --paged_kv=1 --page_size=16 --num_splits=0`
  - Result for both:
    - deterministic all-zero output
    - same reference failure:
      - `max_abs=0.250000`
      - `max_rel=1.000000`
      - `mean_abs=0.025934`
- Current practical conclusion:
  - The real llama-like FA3 forward path is now reproduced again, not just the trimmed kernel path.
  - The non-crashing `num_splits=0` varlen path still produces deterministic zeros on both paged and non-paged inputs.
  - We can now continue wrong-output analysis on the real path instead of the trimmed one.

## 2026-04-02 FA2 Control And Crafted Inputs

- Narrowed `FA2_ARCHS` in `BUILD.bazel` to `["sm_90a"]` so the FA2 control path only builds the H100 target we are actually using.
- Rebuilt with invocation:
  - `545c5011-cf7c-4383-ad2c-4d3e0375ffa8`
- FA2 real-path non-paged control:
  - `LD_LIBRARY_PATH=... ./bazel-bin/fa2_control_repro --batch=1 --seqlen_q=64 --seqlen_k=64 --num_heads=32 --num_heads_k=8 --head_dim=128 --causal=1 --varlen=1 --paged_kv=0 --num_splits=0 --dump_count=8`
  - Result:
    - `PASS`
    - `max_abs=0.000719`
    - `max_rel=0.003534`
    - `mean_abs=0.000045`
    - sample output is nonzero
- FA2 real-path paged control:
  - same config but `--paged_kv=1 --page_size=16`
  - Result:
    - process exits with code `-1` and no stdout/stderr
  - Status:
    - not investigated yet because the non-paged FA2 control already proves the harness and real varlen path can be correct on this machine
- Added deterministic harness input modes in `tests/fa3_sm90_repro.cc`:
  - `uniform_const_v`
  - `uniform_ramp_v`
  - `single_key_copy`
- `uniform_const_v` is the cleanest correctness probe so far:
  - FA3:
    - `./bazel-bin/fa3_sm90_full_repro --batch=1 --seqlen_q=64 --seqlen_k=64 --num_heads=32 --num_heads_k=8 --head_dim=128 --causal=1 --varlen=1 --paged_kv=0 --num_splits=0 --input_mode=uniform_const_v --dump_count=8`
    - Result:
      - `FAIL`
      - `max_abs=0.125000`
      - `max_rel=1.000000`
      - `mean_abs=0.125000`
      - sample output is all zeros
  - FA2:
    - same config with `./bazel-bin/fa2_control_repro`
    - Result:
      - `PASS`
      - exact constant output `0.125000` in the sample
- `uniform_ramp_v` is also a good structured probe:
  - FA3:
    - returns all zeros and fails reference
  - FA2:
    - passes reference with nonzero ramp/prefix-mean output
    - sample begins `0.050049 0.051025 0.052002 0.052979 0.053955 0.054932 0.055908 0.056885`
- `single_key_copy` was not useful as initially framed:
  - both FA2 and FA3 returned zeros against the current CPU reference for `seqlen_k=1`, so the assumption behind that test needs to be revisited before using it as evidence
- Current practical conclusion:
  - We now have a clean A/B on the real non-paged path:
    - FA2 is correct
    - FA3 deterministically zeros outputs
  - The best crafted correctness inputs so far are:
    - `uniform_const_v` for an exact constant-output oracle
    - `uniform_ramp_v` for a structured nontrivial oracle

## 2026-04-02 Untouched Buffer Check

- Added `FA3_REPRO_DEBUG`-guarded sentinel initialization and dumps for:
  - `out`
  - `softmax_lse`
- Rebuilt with invocation:
  - `b2c44bcd-7273-46af-90ea-d2f33c0210ae`
- Real-path non-paged exact oracle:
  - `FA3_REPRO_DEBUG=1 LD_LIBRARY_PATH=... ./bazel-bin/fa3_sm90_full_repro --batch=1 --seqlen_q=64 --seqlen_k=64 --num_heads=32 --num_heads_k=8 --head_dim=128 --causal=1 --varlen=1 --paged_kv=0 --num_splits=0 --input_mode=uniform_const_v --dump_count=8`
  - Sentinels:
    - `out = -0.5`
    - `softmax_lse = -123`
  - Result:
    - `sched 0 1`
    - sampled `out` remains `-0.5`
    - sampled `softmax_lse` remains `-123`
  - Conclusion:
    - clang-built FA3 is not computing zero and storing it on this path
    - it is leaving both output and LSE buffers untouched
- Removed GQA / `pack_gqa` as a cause:
  - same oracle but `--num_heads=32 --num_heads_k=32`
  - `pack_gqa=0`
  - result is still untouched sentinels in both `out` and `softmax_lse`
- Dense nearby case changes symptom instead of fixing correctness:
  - same oracle but `--varlen=0 --num_heads=32 --num_heads_k=8`
  - result:
    - `CUDA error: unspecified launch failure`
- Smaller varlen non-GQA case also reproduces untouched outputs:
  - `--num_heads=1 --num_heads_k=1 --varlen=1`
  - result:
    - `sched 0 1`
    - `out` and `softmax_lse` both remain at their sentinels
- Current practical conclusion:
  - The clang FA3 divergence is now narrower than “wrong math.”
  - On the real non-paged varlen `num_splits=0` path, the kernel launch sequence completes without runtime error but never writes `out` or `softmax_lse`.
  - This is not caused by GQA packing.
  - The nearest dense path still crashes, so the likely boundary remains the SM90 FA3 varlen launch / execution path under clang.

## 2026-04-02 Hermetic NVCC Build-Path Attempt

- Goal:
  - create a reversible `nvcc` A/B path for `//:fa3_sm90_full_repro` without relying on `/usr/local/cuda`
- Checkpoint before the experiment:
  - `58a633f` `Checkpoint before nvcc build-path experiment`
- Switched this branch from the LLVM `cuda_library` macro to `rules_cuda`:
  - `BUILD.bazel` now loads `@rules_cuda//cuda:defs.bzl`
  - `MODULE.bazel` now uses a hermetic `cuda.redist_json(version = "13.0.2")` plus `cuda.toolkit(name = "cuda")`
  - remapped CUDA labels from the old `@cuda//cuda:*` layout to the `rules_cuda` root-repo layout
  - routed around the broken aggregate `@cuda//:cuda_headers` target by depending on specific components:
    - `cudart_headers`
    - `nvcc_headers`
    - `nvvm_headers`
    - `cccl_headers`
    - `crt_headers`
- Useful build results:
  - `a4d3117d-466b-4347-bc70-33f54bbdb7aa`
    - first `rules_cuda` load failed inside `cuda_library.bzl`
  - `f06605b7-0cbb-4051-bac3-b19be30041cb`
    - hermetic repo shape fixed enough to analyze, then failed on broken aggregate `@cuda//:cuda_headers`
  - `5c90d374-fca0-447e-97d0-403521161e38`
    - got into real compilation, then host compile failed until `crt_headers` were added
  - `c1ac02cd-3ee6-4f48-8dd4-686b5a75d442`
    - `bazel` server hit native-thread OOM under `--jobs=1000`
  - `16b05a30-3167-437c-8abe-9617752e42b4`
    - reran in `--batch` mode with the same `--jobs=1000`
    - reached real `nvcc` compilation
    - failed with the first actual nvcc/compiler-boundary error
- Current nvcc blocker:
  - `nvcc` is using the registered LLVM host toolchain
  - failure from `host_config.h`:
    - unsupported clang version (`clang` must be `< 21`)
    - `libc++ is not supported on x86 system`
- Current practical conclusion:
  - hermetic `rules_cuda` is viable enough to reach actual `nvcc` compilation on this branch
  - the remaining issue is not CUDA package discovery
  - the remaining issue is host-toolchain selection: `nvcc` is picking the LLVM/clang/libc++ toolchain instead of a GCC/libstdc++ host compiler

## 2026-04-02 System Clang Host-Compiler Attempt

- Installed system `clang` from `apt` as suggested.
- Disabled `register_toolchains("@llvm//toolchain:all")` in `MODULE.bazel` so the nvcc path would stop picking the LLVM toolchain.
- Rebuilt with:
  - `bazel --batch build --config remote --jobs=1000 //:fa3_sm90_full_repro`
  - invocation `fe75461a-04b2-4dbe-9f71-797dd670e5ef`
- Result:
  - analysis now fails immediately with:
    - `No matching toolchains found for @@bazel_tools//tools/cpp:toolchain_type`
- Conclusion:
  - disabling `@llvm` registration removed the only visible Bazel C++ toolchain on this branch
  - installing system `clang` is not enough by itself; Bazel still needs an explicitly registered non-LLVM C++ toolchain so nvcc can use it as the host compiler
