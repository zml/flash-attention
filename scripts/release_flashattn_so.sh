#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/release_flashattn_so.sh --tag vX.Y.Z [--target <git-ref>] [--notes-file <path>] [--dist-dir <path>] [--no-push-tag]

Builds both flashattn shared-library artifacts locally, creates the git tag, pushes the tag,
and creates or updates the GitHub release with the built artifacts.
EOF
}

tag=""
target="HEAD"
dist_dir="dist/release"
notes_file=""
push_tag=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      tag="${2:-}"
      shift 2
      ;;
    --target)
      target="${2:-}"
      shift 2
      ;;
    --notes-file)
      notes_file="${2:-}"
      shift 2
      ;;
    --dist-dir)
      dist_dir="${2:-}"
      shift 2
      ;;
    --no-push-tag)
      push_tag=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$tag" ]]; then
  usage >&2
  exit 1
fi

case "$tag" in
  v*)
    ;;
  *)
    echo "tag must match v*" >&2
    exit 1
    ;;
esac

if ! git rev-parse --verify "$target^{commit}" >/dev/null 2>&1; then
  echo "target is not a valid commit-ish: $target" >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "gh is not authenticated" >&2
  exit 1
fi

requested_target="$(git rev-parse "$target^{commit}")"
head_target="$(git rev-parse HEAD^{commit})"
if [[ "$requested_target" != "$head_target" ]]; then
  echo "checked-out HEAD ($head_target) does not match --target ($requested_target)" >&2
  echo "checkout the target commit locally first, then rerun" >&2
  exit 1
fi

mkdir -p "$dist_dir"
amd64_artifact="$dist_dir/libflashattn-linux-amd64.so"
arm64_artifact="$dist_dir/libflashattn-linux-arm64.so"

bazel build \
  -c opt \
  :flashattn_so_for_all_platforms \
  --config remote

cp bazel-out/linux_amd64-opt/bin/libflashattn.so "$amd64_artifact"
cp bazel-out/linux_arm64-opt/bin/libflashattn.so "$arm64_artifact"

if git rev-parse --verify "refs/tags/$tag" >/dev/null 2>&1; then
  existing_target="$(git rev-list -n 1 "$tag")"
  if [[ "$existing_target" != "$requested_target" ]]; then
    echo "tag $tag already exists on $existing_target, not $requested_target" >&2
    exit 1
  fi
else
  git tag "$tag" "$target"
fi

if [[ "$push_tag" -eq 1 ]]; then
  git push origin "refs/tags/$tag"
fi

release_args=(
  "$tag"
  "$amd64_artifact"
  "$arm64_artifact"
)

if [[ -n "$notes_file" ]]; then
  release_args+=(--notes-file "$notes_file")
else
  release_args+=(--generate-notes)
fi

if gh release view "$tag" >/dev/null 2>&1; then
  gh release upload "$tag" "$amd64_artifact" "$arm64_artifact" --clobber
else
  gh release create "${release_args[@]}"
fi
