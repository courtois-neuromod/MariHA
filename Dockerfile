FROM nvidia/cuda:12.5.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Python 3.12 via deadsnakes
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev \
        build-essential cmake git zlib1g-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Install cuDNN 9.3.x via pip to exactly match TF 2.21's build requirements.
# The nvidia/cuda base image ships cuDNN 9.2.x; this overrides it.
RUN pip install --no-cache-dir "nvidia-cudnn-cu12>=9.3,<9.4"

# Point LD_LIBRARY_PATH to pip-installed cuDNN so TF finds 9.3.x at runtime
ENV CUDNN_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn
ENV LD_LIBRARY_PATH=${CUDNN_PATH}/lib:${LD_LIBRARY_PATH}

# All other deps — plain tensorflow (CUDA runtime from base image, cuDNN from above)
RUN pip install --no-cache-dir \
    "tensorflow>=2.16" \
    "tf_keras>=2.13" \
    "tensorflow-probability>=0.21" \
    "stable-retro>=0.9.2" \
    "opencv-python-headless>=4.8" \
    "gymnasium>=0.29" \
    "numpy>=1.24" \
    "pandas>=2.0" \
    "tensorboard>=2.13" \
    "tqdm>=4.65" \
    "rich>=13.0"

WORKDIR /app
CMD ["bash"]
