# Build environment for V8 8.5 (mid-2020). An x64 Ubuntu 20.04 image matches the
# toolchain that release expects: its DEPS hooks still call `python` (2.x), and its
# prebuilt clang only exists for x64 Linux. On Apple Silicon this runs under Rosetta.
FROM --platform=linux/amd64 ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -yqq \
        bison \
        cdbs \
        curl \
        file \
        flex \
        g++ \
        git \
        lsb-release \
        ninja-build \
        pkg-config \
        python-is-python2 \
        python3 \
        python3-pip \
        sudo \
        vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /v8
CMD ["/v8/build.sh"]
