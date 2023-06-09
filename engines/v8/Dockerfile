FROM debian:stable-slim as builder

RUN apt-get update && apt-get upgrade -yqq

RUN DEBIAN_FRONTEND=noninteractive \
    apt-get install bison \
                    cdbs \
                    curl \
                    flex \
                    g++ \
                    git \
                    python \
                    vim \
                    pkg-config -yqq \
                    python3 \
                    ninja-build