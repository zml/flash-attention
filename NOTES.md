# NOTES

## Goal

Build a small, compiler-controlled reproducer for the FA3 Hopper clang failure mode, with the immediate focus on:

- out-of-line GMMA helper calls
- `ptxas` `C7510` warnings about WGMMA crossing function boundaries
- large stack / local-memory traffic in clang-built device code

The working hypothesis is no longer "general FA3 runtime repro first." The current strategy is to isolate the codegen shape that makes clang produce the bad kernel while nvcc does not.

## Current Strategy

1. Use single-TU Bazel targets only.
2. Compare clang vs nvcc from clean builds with copied object snapshots.
3. Inspect the actual device payload, not just host objects:
   - extracted fatbins
   - extracted cubins
   - PTX when present
   - SASS / resource usage / symbol tables
4. Build progressively smaller repros that stay as close as possible to the FA3 mainloop shape.
5. Use FA2 as the known-good control.

## Current Facts

- The strongest bad signal is still the reduced FA3 TU:
  - `hopper/instantiations/flash_fwd_hdim128_bf16_paged_split_sm90.cu`
- For that TU, clang and nvcc produce materially different device code.
- The important divergence is not entrypoint naming. It is code shape:
  - clang keeps much larger device payloads
  - clang keeps extra helper/device symbols
  - clang produces the suspicious WGMMA / function-boundary behavior
- FA2 remains the control case:
  - clang and nvcc still differ in object/cubin size and helper-symbol clutter
  - but the actual device entrypoint names match
  - and we have not seen the FA3-style GMMA spill/call pathology there

## FA3 Reduced-TU Findings Driving This Strategy

- Clean comparison target:
  - `//:flashattn-sm90-llama-aquery`
- Compared TU:
  - `hopper/instantiations/flash_fwd_hdim128_bf16_paged_split_sm90.cu`
- Known bad clang-vs-nvcc device-code deltas for that TU:
  - clang has much larger cubin / fatbin payload
  - clang has large `STACK` and heavy local-memory traffic
  - clang has many extra device calls
  - clang PTX / `ptxas` show WGMMA crossing function boundaries
- This is the shape we are trying to reproduce in a smaller source file.

## Standalone Reproducer Attempts

### Attempt 1

- File:
  - `hopper/clang_gmma_boundary_repro.cu`
- Shape:
  - standalone Hopper WGMMA kernel
  - shared-memory A/B tiles
  - helper chain `gmma_mainloop -> gmma_step -> gmma_wrapper -> gmma_leaf`
- Result:
  - builds under both clang and nvcc
  - clang still embeds PTX, nvcc does not
  - but no FA3-style bad code shape appears
  - no obvious `CALL`, `CALL.ABS`, `STL`, or `LDL` inflation

### Attempt 2

- Same file, refined toward FA3:
  - nested lambdas inside `gmma_wrapper`
  - local `gmma(...)` helper modeled on Hopper `flash::gemm`
  - local `convert_layout_acc_Aregs(...)`
  - local `convert_type_out(...)`
- Result:
  - still builds under both clang and nvcc
  - still does not reproduce the FA3 spill / call pattern

### Attempt 3

- Same file:
  - `hopper/clang_gmma_boundary_repro.cu`
- New shape:
  - pulled the standalone toy closer to the `fwd_step` pressure pattern
  - keeps outer live state across the closure:
    - `tOrO`
    - `tOrP`
    - small `scores_scale`
  - does conversion inside the closure:
    - `convert_layout_acc_Aregs(...)`
    - `convert_type_out(...)`
  - adds outer helper lambdas:
    - `scoremod`
    - `rescale_o`
    - `finalize_dispatch`
    - `fwd_step`
- Build results:
  - clang:
    - BuildBuddy: `261cceb3-101e-40ac-b4bb-8e9497c0d5b5`
  - nvcc:
    - BuildBuddy: `86e64d53-ad33-4843-a7d3-b9f11c2492c4`
- New copied snapshots:
  - clang:
    - `/tmp/gmma_repro_compare_v3/clang.o`
    - `/tmp/gmma_repro_compare_v3/clang.pic.o`
    - `.o == .pic.o`
    - sha256: `569fde9f7aae2979a39950f6577d0eb48520f4dc9e39ad790d8a656bbca682fa`
  - nvcc:
    - `/tmp/gmma_repro_compare_v3/nvcc.o`
    - `/tmp/gmma_repro_compare_v3/nvcc.pic.o`
    - non-PIC sha256: `683df66a4af118e6e6daa6485b1febf84865b914bc729e3199bdb9f2867c4a4f`
    - PIC sha256: `9c970eaee7f3f27ba34c626c233534ea0f82431b13cc7401560bba27ecc10543`
- Important new result:
  - this source is now enough to trigger a real WGMMA pipeline warning in the standalone reproducer
  - both clang and nvcc emit:
    - `C7517`
    - `warpgroup.wait is injected ... to allow use of registers defined by GMMA`
- Resource / code-shape result:
  - still no clang-only spill or call explosion
  - clang:
    - `REG:96 STACK:0 SHARED:1024 GLOBAL:0`
  - nvcc:
    - `REG:96 STACK:0 SHARED:1024 GLOBAL:12 CONSTANT[4]:96`
  - still no obvious `CALL`, `CALL.ABS`, `STL`, or `LDL` in either SASS dump
- Interpretation:
  - moving live state and the QK-to-`tOrP` conversion into the closure was enough to create a real WGMMA hazard
  - but it is still not enough to reproduce the FA3 clang-only pathological lowering
  - the next missing ingredient is likely the real PV handoff and/or pipeline/barrier state rather than just closure pressure

### Attempt 4

- Same file:
  - `hopper/clang_gmma_boundary_repro.cu`
- New shape:
  - kept the closure-heavy `fwd_step` structure
  - replaced the fake `axpby` consumer with a real second GMMA handoff
  - added separate shared-memory `sV`
  - built a PV-style `tiled_mma_pv`
  - converted `tSrS -> tOrP_acc -> tOrP`
  - fed `tOrP` into a second GMMA to produce `tOrO`
- Build results:
  - clang:
    - BuildBuddy: `74a0f8e7-7593-4a77-a501-0eb0cad1b406`
  - nvcc:
    - BuildBuddy: `5975ce5a-3a42-4b0f-84db-48de823a7241`
- Clean copied snapshots:
  - clang:
    - `/tmp/gmma_repro_compare_v4/clang.o`
    - `/tmp/gmma_repro_compare_v4/clang.pic.o`
    - `.o == .pic.o`
    - sha256: `619082c5feba138aa2021b374f1bcabd7eebc567036bddcc8e45dc533928c4cd`
  - nvcc:
    - `/tmp/gmma_repro_compare_v4/nvcc.o`
    - `/tmp/gmma_repro_compare_v4/nvcc.pic.o`
    - non-PIC sha256: `b63678b61272fdb87aeb7a5979a077e9837db31990fe59eb38e48e951b7ace5f`
    - PIC sha256: `6bdd2034142d6e257ba54848baba6f18231c0eb3785f33cc9515ec7a46d7f371`
- Important new result:
  - this is the first standalone reproducer that cleanly separates clang from nvcc in the same way as the FA3 reduced TU
  - clang emits:
    - many `C7519`
    - many `C7517`
    - `C7510` `wgmma.mma_async instructions are serialized due to wgmma pipeline crossing function boundary`
  - nvcc emits:
    - `C7517`
    - `C7511` `insufficient register resources for the wgmma pipeline`
    - no `C7510`
- Resource / code-shape result:
  - clang:
    - `REG:128 STACK:928 SHARED:1024 GLOBAL:0`
    - embedded PTX present
    - PTX entry explicitly allocates:
      - `.local .align 16 .b8 __local_depot0[928];`
    - PTX entry contains two `call.uni` sites into outlined `gmma_leaf` helper bodies
    - SASS counts:
      - `CALL.REL:2`
      - `STL:780`
      - `LDL:32`
  - nvcc:
    - `REG:218 STACK:0 SHARED:1024 GLOBAL:12 CONSTANT[4]:96`
    - no embedded PTX
    - no `CALL`, `STL`, or `LDL` found in the SASS dump
- Key interpretation:
  - the reproducer now matches the essential clang failure signature:
    - clang outlines GMMA-adjacent helper code
    - clang pays for that with a real per-thread stack frame and local-memory traffic
    - nvcc compiles the same source by keeping everything in registers and staying stack-free
  - this makes the issue look less like a missing frontend compatibility flag and more like a clang CUDA lowering / outlining choice in the two-stage GMMA handoff pattern

## Current Interpretation

Nested lambdas plus a local WGMMA helper are not enough on their own.

Attempt 4 shows that the crucial missing ingredient was the real second-GMMA handoff, not full FA3 pipeline state.

The current best local theory is:

- clang outlines the two-stage GMMA helper path into callable device functions
- that creates a function-boundary WGMMA pipeline break
- that in turn introduces a real stack frame and local-memory traffic in the entry kernel
- nvcc instead accepts much higher register usage and keeps the path inline enough to stay stack-free

The remaining question is whether any clang flag can force the nvcc-like choice, or whether this is simply a clang codegen bug in this source shape.

Additional FA3-specific state may still matter, especially:

- real pipeline/barrier flow
- `BlockMN_t::get_n_block_min_max(...)`
- the exact `CollectiveMainloopFwdSm90::mma(...)` closure shape
- the interaction between QK and PV paths in the same function body under the real scheduler/state objects

## FA2 Control

- Added single-TU control target:
  - `//:flashattn-fa2-single-aquery`
- TU:
  - `csrc/flash_attn/src/flash_fwd_hdim64_bf16_sm80.cu`
- Reason:
  - FA2 is known-good on both clang and nvcc
  - compare symbol/cubin differences without Hopper GMMA complications

### FA2 device-symbol result

- Extracted cubins:
  - `/tmp/fa2_single_compare/dumps/clang.pic.1.sm_90a.cubin`
  - `/tmp/fa2_single_compare/dumps/nvcc.pic.1.sm_90a.cubin`
- Result:
  - actual device kernel entrypoint mangling matches between clang and nvcc
  - clang still carries extra weak helper symbols and a larger cubin
  - but not the FA3-style pathological code shape

## Immediate Next Step

Build the next standalone reproducer by mirroring a thinner but more faithful slice of `CollectiveMainloopFwdSm90::mma(...)`:

- preserve the nested `fwd_step` closure structure
- include both QK and PV-style GMMA calls in one function body
- keep the minimal amount of pipeline / barrier / mask state needed to trigger the same lowering decisions

Acceptance criteria for a useful reproducer:

- clang emits out-of-line helper calls or `call.uni` around GMMA while nvcc does not, or
- clang shows the same `STACK` / `STL` / `LDL` blow-up while nvcc stays flat, or
- clang reproduces `ptxas` `C7510` WGMMA-boundary warnings on the small TU

The current standalone repro now satisfies all three.

## Immediate Next Step

Use this small reproducer to test whether any clang-side flag can remove the bad lowering shape without changing the source:

- compare exact clang vs nvcc command lines for this TU
- try clang-side optimization/inlining deltas only
- accept only changes that remove:
  - `C7510`
  - `call.uni` / `CALL.REL`
  - the `928` byte local stack

## New Strategy

We are now splitting the work into two tracks, with the flag track first.

### Track A: clang flag experiments on the current repro

Use the current `hopper/clang_gmma_boundary_repro.cu` as the baseline because it already reproduces:

- clang `C7510`
- outlined helper calls
- `STACK:928`
- heavy `STL` / `LDL`
- nvcc staying stack-free on the same source

The goal is to test whether any clang-side codegen or inlining flag can force a more nvcc-like lowering without changing the source.

Acceptance criteria for a promising flag variant:

- `C7510` disappears
- `call.uni` / `CALL.REL` disappear
- stack collapses substantially or reaches zero

### First flag sweep

Added clang-only comparison targets in `BUILD.bazel`:

- `//:clang-gmma-boundary-repro`
- `//:clang-gmma-boundary-repro-o3`
- `//:clang-gmma-boundary-repro-inline`
- `//:clang-gmma-boundary-repro-nordc`

The first sweep was run from clean builds and copied snapshots were written to:

- `/tmp/gmma_flag_sweep/`

Tested flag sets:

- baseline:
  - no extra codegen flags beyond the original repro target
- `-O3`:
  - just add `-O3`
- inline bundle:
  - `-O3`
  - `-fgpu-inline-threshold=100000`
  - `-finline-functions`
  - `-finline-hint-functions`
  - `-finline-max-stacksize=4096`
  - `-mllvm -inline-threshold=100000`
- explicit no-RDC:
  - `-O3`
  - `-fno-gpu-rdc`

Results:

- baseline:
  - still emits `C7510`
  - `REG:128 STACK:928`
  - `CALL.REL:2`
  - `STL:780`
  - `LDL:32`
  - `call.uni:2`
- `-O3`:
  - identical to baseline
- `-fno-gpu-rdc`:
  - identical to baseline
- inline bundle:
  - `C7510` disappears
  - warning changes to `C7511`, matching nvcc's warning class
  - `REG:218 STACK:0`
  - `CALL.REL:0`
  - `STL:0`
  - `LDL:0`
  - `call.uni:0`

Interpretation:

- `-O3` alone is not enough
- explicit `-fno-gpu-rdc` is irrelevant here
- the bad clang behavior is highly sensitive to inlining thresholds and stack-budget heuristics
- the inline bundle is the first concrete flag-based workaround candidate because it forces clang into the same high-register / zero-stack shape that nvcc already chooses

### Track B: monotonic source minimization

Source minimization is still worthwhile, but only if the reduced source preserves the exact bad signal.

This means reducing one ingredient at a time and only keeping variants that still show:

- clang `C7510`
- clang helper outlining / calls
- clang nonzero stack and local-memory traffic
- nvcc no `C7510`

The current plan is to defer this until after the first round of flag experiments, to avoid accidentally minimizing away the critical trigger before we test for a usable workaround.

## Historical Notes

The previous running log has been preserved in:

- `NOTESOLD.md`
