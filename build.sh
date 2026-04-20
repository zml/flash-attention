#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

image_tag="${FLASHATTN_DOCKER_IMAGE:-flashattn-builder:latest}"
container_name="${FLASHATTN_DOCKER_CONTAINER:-flashattn-builder-extract-$$}"
artifact_path="${FLASHATTN_ARTIFACT_PATH:-${repo_root}/bazel-bin/libflashattn.so}"
container_workspace="/flashattn"
container_artifact_path="${container_workspace}/bazel-bin/libflashattn.so"

cleanup() {
    docker rm -f "${container_name}" >/dev/null 2>&1 || true
}

trap cleanup EXIT

mkdir -p -- "$(dirname -- "${artifact_path}")"

docker build -t "${image_tag}" "${repo_root}"
docker create \
    --name "${container_name}" \
    -v "${repo_root}:${container_workspace}" \
    "${image_tag}" >/dev/null

docker start -a "${container_name}"
docker cp -L "${container_name}:${container_artifact_path}" "${artifact_path}"

printf 'Extracted %s\n' "${artifact_path}"
