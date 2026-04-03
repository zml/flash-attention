# AGENTS

This branch is being used to isolate an FA3 forward-pass correctness bug that shows up in a custom Llama path while FA2 remains correct.

Primary working notes live in `NOTES.md`.

Working rules for this branch:
- Update `NOTES.md` after each meaningful experiment.
- Commit local checkpoints often.
- Push after each checkpoint commit because the machine is preemptible.
- Prefer small, reversible repro-focused changes over broad refactors.

Bootstrap / build setup:
- Run `./bootstrap.sh` on a fresh machine before trying the `rules_cuda` + clang path. It installs the required host packages, downloads the bootstrapped LLVM toolchain, replaces `clang` / `clang++` symlinks with in-install hardlinks, and initializes the CUDA submodules used by this branch.
- For the local `rules_cuda` + clang build, use:
  - `bazel build --jobs=64 --repo_env=BAZEL_DO_NOT_DETECT_CPP_TOOLCHAIN=0 --repo_env=CUDA_CLANG_PATH=$HOME/.cache/llvm-toolchain/bin/clang --repo_env=CC=$HOME/.cache/llvm-toolchain/bin/clang --repo_env=CXX=$HOME/.cache/llvm-toolchain/bin/clang++ --@rules_cuda//cuda:compiler=clang //:fa3_sm90_full_repro`
