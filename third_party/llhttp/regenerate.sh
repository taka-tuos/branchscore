#!/bin/sh
set -eu

root_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
upstream_dir="$root_dir/upstream"
generated_dir="$root_dir/generated"

if ! command -v npm >/dev/null 2>&1; then
    echo "error: npm is required to regenerate vendored llhttp sources" >&2
    exit 1
fi

cleanup() {
    rm -rf "$upstream_dir/node_modules" "$upstream_dir/build"
}
trap cleanup EXIT HUP INT TERM

npm --prefix "$upstream_dir" ci --ignore-scripts
npm --prefix "$upstream_dir" run build

install -D -m 0644 "$upstream_dir/build/c/llhttp.c" \
    "$generated_dir/src/llhttp.c"
install -D -m 0644 "$upstream_dir/build/llhttp.h" \
    "$generated_dir/include/llhttp.h"

echo "Regenerated llhttp 9.3.1 sources under $generated_dir"
