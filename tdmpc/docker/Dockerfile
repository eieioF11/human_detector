FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    software-properties-common \
    wget \
    xauth \
    x11-apps \
    libclang-dev \
    libatk-bridge2.0 \
    libfontconfig1-dev \
    libfreetype6-dev \
    libgl1-mesa-dev \
    libglib2.0-dev \
    libgtk-3-dev \
    libssl-dev \
    libxcb-render0-dev \
    libxcb-shape0-dev \
    libxcb-xfixes0-dev \
    libxkbcommon-dev \
    libxkbcommon-x11-dev \
    patchelf \
    vulkan-tools \
    git \
    build-essential \
    swig \
    fontconfig \
    htop \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python (version 3.10)
RUN add-apt-repository ppa:deadsnakes/ppa && \
  apt-get update && apt-get install -y \
  python3.10 \
  python3.10-dev \
  python3.10-venv \
  python3.10-distutils \
  python3.10-tk

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies
WORKDIR /workspace
COPY README.md .
COPY pyproject.toml .
COPY uv.lock .
COPY .python-version .
RUN uv sync --frozen --no-editable

# # -------------------- Create non-root user --------------------
#   ARG USERNAME=vscode
#   ARG USER_UID=1000
#   RUN useradd -m -s /bin/bash -u ${USER_UID} ${USERNAME}
#   USER ${USERNAME}
#   WORKDIR /workspace
  
#   # -------------------- Install deps in *non-editable* mode ----
#   COPY --chown=${USERNAME}:${USERNAME} pyproject.toml uv.lock README.md ./
#   COPY --chown=${USERNAME}:${USERNAME} src/ src/
#   RUN uv sync --frozen 
  
#   # -------------------- Default shell --------------------
#   CMD ["bash"]