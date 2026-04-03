#!/usr/bin/env bash
set -euo pipefail

LLVM_VERSION="22.1.0-3"
LLVM_ARCHIVE="llvm-toolchain-minimal-22.1.0-linux-amd64-musl.tar.zst"
LLVM_URL="https://github.com/cerisier/toolchains_llvm_bootstrapped/releases/download/llvm-${LLVM_VERSION}/${LLVM_ARCHIVE}"
LLVM_ROOT="${HOME}/.cache/llvm-toolchain"

echo "[bootstrap] installing required host packages"
sudo apt-get update
sudo apt-get install -y \
  curl \
  zstd \
  g++-12 \
  libstdc++-12-dev

echo "[bootstrap] downloading llvm toolchain if needed"
mkdir -p "${HOME}/.cache"
if [[ ! -x "${LLVM_ROOT}/bin/llvm" ]]; then
  rm -rf "${LLVM_ROOT}"
  mkdir -p "${LLVM_ROOT}"
  curl -fL --retry 3 -o "${HOME}/.cache/${LLVM_ARCHIVE}" "${LLVM_URL}"
  tar --zstd -xf "${HOME}/.cache/${LLVM_ARCHIVE}" -C "${LLVM_ROOT}" --strip-components=0
fi

echo "[bootstrap] replacing clang symlinks with hardlinks inside the llvm install"
rm -f "${LLVM_ROOT}/bin/clang" "${LLVM_ROOT}/bin/clang++"
ln "${LLVM_ROOT}/bin/llvm" "${LLVM_ROOT}/bin/clang"
ln "${LLVM_ROOT}/bin/llvm" "${LLVM_ROOT}/bin/clang++"

echo "[bootstrap] ensuring flash-attention submodules are present"
git submodule update --init --recursive csrc/cutlass csrc/composable_kernel

cat <<EOF

[bootstrap] done

Use this Bazel invocation for rules_cuda + clang:

bazel build --jobs=64 \\
  --repo_env=BAZEL_DO_NOT_DETECT_CPP_TOOLCHAIN=0 \\
  --repo_env=CUDA_CLANG_PATH=${LLVM_ROOT}/bin/clang \\
  --repo_env=CC=${LLVM_ROOT}/bin/clang \\
  --repo_env=CXX=${LLVM_ROOT}/bin/clang++ \\
  --@rules_cuda//cuda:compiler=clang \\
  //:fa3_sm90_full_repro

EOF
