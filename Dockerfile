FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG BAZELISK_VERSION=v1.25.0
ARG TARGETARCH=amd64

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

WORKDIR /flashattn

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /usr/local/bin/bazel \
    https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-${TARGETARCH} \
    && chmod +x /usr/local/bin/bazel

COPY .bazelversion .

RUN USE_BAZEL_VERSION=$(cat .bazelversion) bazel version

ENTRYPOINT ["bazel"]
CMD ["build", "--config", "docker", "-c", "opt", "//:flashattn_so"]
