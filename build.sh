#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(pwd)"

host_arch="$(uname -m)"

# Cross-compilation is not practically feasible with rules_cuda at the moment.
# So compilation needs to be done from a machine with the same architecture as the target.
# We deduce it here.
case "${host_arch}" in
    aarch64|arm64)
        docker_targetarch="arm64"
        cuda_platform="linux-sbsa"
        ;;
    x86_64|amd64)
        docker_targetarch="amd64"
        cuda_platform="linux-x86_64"
        ;;
    *)
        printf 'Unsupported host architecture: %s\n' "${host_arch}" >&2
        exit 1
        ;;
esac

printf 'Building for architecture: %s\n' "${docker_targetarch}"

image_tag="${FLASHATTN_DOCKER_IMAGE:-flashattn-builder:latest}"
container_name="${FLASHATTN_DOCKER_CONTAINER:-flashattn-builder-extract-$$}"
artifact_path="${FLASHATTN_ARTIFACT_PATH:-${repo_root}/libflashattn.so}"
container_workspace="/flashattn"
container_artifact_path="${container_workspace}/bazel-bin/libflashattn.so"

cleanup() {
    docker rm -f "${container_name}" >/dev/null 2>&1 || true
}

trap cleanup EXIT

mkdir -p -- "$(dirname -- "${artifact_path}")"

docker build \
    --build-arg TARGETARCH="${docker_targetarch}" \
    -t "${image_tag}" \
    "${repo_root}"
docker create \
    --name "${container_name}" \
    -v "${repo_root}:${container_workspace}" \
    "${image_tag}" \
    build \
    --config docker \
    -c opt \
    //:flashattn_so \
    --@rules_cuda//cuda:exec_platform="${cuda_platform}" \
    --@rules_cuda//cuda:target_platform="${cuda_platform}" \
    >/dev/null

docker start -a "${container_name}"
docker cp -L "${container_name}:${container_artifact_path}" "${artifact_path}"

printf 'Extracted %s\n' "${artifact_path}"
