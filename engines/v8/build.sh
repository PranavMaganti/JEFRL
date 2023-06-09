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
git checkout d8217c3257d905a6e32e28f7c7ac6328a1be343f
gclient sync

# Build d8
gn gen out/fuzzbuild --args='is_debug=false dcheck_always_on=true v8_static_library=true v8_enable_slow_dchecks=true v8_enable_v8_checks=true v8_enable_verify_heap=true v8_enable_verify_csa=true v8_fuzzilli=true v8_enable_verify_predictable=true sanitizer_coverage_flags="trace-pc-guard" target_cpu="x64"'
ninja -C ./out/fuzzbuild d8