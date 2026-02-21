ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.12-py3
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# See https://github.com/openucx/ucc/issues/476#issuecomment-1766207267
ENV LD_LIBRARY_PATH="/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH"

ADD . /workspace
WORKDIR /workspace

RUN apt update && \
    apt install -y rsync ssh software-properties-common

RUN UBUNTU_VERSION=$(grep VERSION_ID /etc/os-release | cut -d '"' -f 2 | tr -d '.') && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt update && \
    apt -y install cudss-cuda-12 libcudss0-dev-cuda-12

RUN add-apt-repository -y ppa:fenics-packages/fenics && \
    apt update && \
    apt install -y fenicsx

RUN pip install -r /workspace/container_src/requirements.txt

RUN git clone --recursive https://github.com/NVIDIA/AMGX && \
    cd AMGX && \
    mkdir build && \
    cd build && \
    cmake .. -DCUDA_ARCH="80;90" && \
    make -j 10 all

RUN AMGX_DIR=/workspace/AMGX \
    pip install git+https://github.com/mgr0dzicki/pyamgx.git@allow_other_data_types

RUN pip install --no-build-isolation -e /workspace/src/dd_solvers

RUN pip install -e /workspace/src/experiments

RUN chmod +x /workspace/container_src/start.sh
CMD ["/workspace/container_src/start.sh"]
