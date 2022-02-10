#FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   nano \
                   ca-certificates \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    torch

#RUN git clone https://github.com/NVIDIA/apex
#RUN cd apex && \
#    python3 setup.py install && \
#    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

WORKDIR /workspace
COPY . unetr-heart/
RUN cd unetr-heart/

#    && \
#    python3 -m pip install --no-cache-dir .
# CMD ["/bin/bash"]