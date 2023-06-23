#! /bin/bash

cd v8

if [ ! -d "depot_tools" ] ; then
    git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git
fi
export PATH="/v8/depot_tools:${PATH}"

# Fetch v8
fetch v8

# Checkout v8 version and update dependencies
cd v8
git checkout 8.5-lkgr
gclient sync

# Build d8
gn gen out/fuzzbuild-latest --args='is_debug=false dcheck_always_on=true v8_static_library
=true v8_enable_verify_heap=true v8_fuzzilli=true sanitizer_coverage_flags="trace-pc-guard" target_cpu="x64"is_asan=true is_lsan=true is_ubsan=true is_ubsan_no_recover=true'
ninja -C ./out/fuzzbuild-8.5 d8