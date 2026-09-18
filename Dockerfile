FROM ubuntu:24.04

# 24.04 ships python3.12; other versions come from deadsnakes
ARG python=3.12

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PATH="/root/.cargo/bin:$PATH" \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    TZ=Etc/UTC \
    BLIS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    NUMBA_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    VECLIB_MAXIMUM_THREADS=2

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && \
    apt-get update && \
    apt-get install --no-install-recommends -y \
        apt-transport-https \
        dirmngr \
        dpkg-dev \
        gpg-agent \
        locales \
        software-properties-common \
        gnupg2 \
        wget && \
    locale-gen $LC_ALL && \
    update-locale && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    wget -q https://packages.microsoft.com/config/ubuntu/$(. /etc/os-release && echo $VERSION_ID)/packages-microsoft-prod.deb -O packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    wget -qO- https://sh.rustup.rs | sh -s -- --no-modify-path --default-toolchain stable -y && \
    apt-get update && \
    apt-get install --no-install-recommends -y \
        dotnet-sdk-8.0 \
        g++ \
        gcc \
        git \
        nodejs \
        openjdk-17-jdk-headless \
        php \
        powershell \
        python${python}-dev \
        python3-setuptools && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /m2cgen

COPY requirements-test.txt ./
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${python} 1 && \
    wget -qO- https://bootstrap.pypa.io/get-pip.py | python && \
    pip install --no-cache-dir -r requirements-test.txt
