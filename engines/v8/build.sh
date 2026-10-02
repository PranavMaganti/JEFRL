#!/bin/bash
# Builds d8 with Fuzzilli's coverage instrumentation and REPRL support
# (v8_fuzzilli=true). Run inside the Docker image in this folder, see the README.
#
#   BUILD=fuzzbuild  (default) the build to fuzz with: DCHECKs on, no sanitizer
#   BUILD=asan       AddressSanitizer build, to triage crashes found by fuzzbuild
#   V8_VERSION       the V8 branch/tag to build (default 8.5-lkgr)
#   JOBS             parallel compile jobs (default: number of CPUs)
set -euo pipefail

V8_VERSION=${V8_VERSION:-8.5-lkgr}
BUILD=${BUILD:-fuzzbuild}
# depot_tools as of the V8 release, as current versions no longer sync old checkouts
DEPOT_TOOLS_DATE=${DEPOT_TOOLS_DATE:-2020-07-15}

cd /v8

if [ ! -d depot_tools ]; then
    git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git
    git -C depot_tools checkout "$(git -C depot_tools rev-list -n 1 --before="$DEPOT_TOOLS_DATE" main)"
fi
export PATH="/v8/depot_tools:${PATH}"
export DEPOT_TOOLS_UPDATE=0

if [ ! -d v8 ]; then
    fetch --nohooks v8
fi

cd v8
git fetch origin "$V8_VERSION"
git checkout FETCH_HEAD
gclient sync -D

# Make d8 flush its output after each REPRL execution, so an execution's output
# (and the exception it throws) isn't read as part of a later one
if ! git apply --reverse --check /v8/reprl-flush.patch 2>/dev/null; then
    git apply /v8/reprl-flush.patch
fi

# Flags from Fuzzilli's V8 fuzzbuild. ASan is kept out of the fuzzing build: it
# makes every execution several times slower, and crashes found without it can be
# re-run on the ASan build to get a report.
COMMON_ARGS='is_debug=false dcheck_always_on=true v8_static_library=true v8_enable_verify_heap=true v8_fuzzilli=true sanitizer_coverage_flags="trace-pc-guard" target_cpu="x64"'

case "$BUILD" in
    fuzzbuild)
        ARGS="$COMMON_ARGS v8_enable_slow_dchecks=true v8_enable_v8_checks=true"
        ;;
    asan)
        ARGS="$COMMON_ARGS is_asan=true"
        ;;
    *)
        echo "Unknown BUILD: $BUILD (expected fuzzbuild or asan)" >&2
        exit 1
        ;;
esac

gn gen "out/$BUILD" --args="$ARGS"
# Lower JOBS if the compiler runs out of memory (roughly 1-2 GB per job)
ninja -C "out/$BUILD" -j "${JOBS:-$(nproc)}" d8

echo "Built $(pwd)/out/$BUILD/d8"
