# The matrix Python versions come from the official images; the language
# toolchains (dotnet for C#/Visual Basic, OpenJDK, Node.js for the
# JavaScript executor, PHP, PowerShell, Rust) from Debian + Microsoft repos.
ARG python=3.12
FROM python:${python}-slim-bookworm

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
        ca-certificates \
        dirmngr \
        gnupg2 \
        locales \
        wget && \
    locale-gen $LC_ALL && \
    update-locale && \
    wget -q https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb -O packages-microsoft-prod.deb && \
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
        powershell && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /m2cgen

COPY requirements-test.txt ./
RUN pip install --no-cache-dir -r requirements-test.txt
