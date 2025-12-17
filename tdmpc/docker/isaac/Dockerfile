# A lightweight Docker image for running Isaac Lab 2.0.0
# Please refer to the official documentation for the officlal docker usages.
FROM nvcr.io/nvidia/isaac-sim:4.5.0

# Install Conda
# Ref: https://github.com/conda-forge/miniforge-images/blob/484f385f409e75a194dbe48633d8662d389ad748/ubuntu/Dockerfile
# > Starting with the 22.11 PyTorch NGC container, miniforge is removed and all Python packages are installed in the default Python environment...
# > See: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-01.html
ARG MINIFORGE_NAME=Miniforge3
ARG MINIFORGE_VERSION=24.3.0-0
ARG TARGETPLATFORM

ENV CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=${CONDA_DIR}/bin:${PATH}

ENV ROS_DISTRO humble
ENV ROS_WS ros2_ws
ARG DEBIAN_FRONTEND=noninteractive
ENV __NV_PRIME_RENDER_OFFLOAD=1
ENV __GLX_VENDOR_LIBRARY_NAME=nvidia

# Install tools
RUN apt-get update && apt-get install -y --allow-downgrades libbrotli1=1.0.9-2build6
RUN apt-get -qq update && apt-get -q -y install \
    gnupg2 \
    gnupg \
    lsb-release \
    curl \
    ca-certificates \
    git \
    gh \
    tmux \
    nano \
    vim \
    wget \
    net-tools \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get -qq clean

# 1. Install just enough for conda to work
# 2. Keep $HOME clean (no .wget-hsts file), since HSTS isn't useful in this context
# 3. Install miniforge from GitHub releases
# 4. Apply some cleanup tips from https://jcrist.github.io/conda-docker-tips.html
#    Particularly, we remove pyc and a files. The default install has no js, we can skip that
# 5. Activate base by default when running as any *non-root* user as well
#    Good security practice requires running most workloads as non-root
#    This makes sure any non-root users created also have base activated
#    for their interactive shells.
# 6. Activate base by default when running as root as well
#    The root user is already created, so won't pick up changes to /etc/skel
RUN apt-get update > /dev/null && \
    apt-get install --no-install-recommends --yes \
    wget bzip2 ca-certificates \
    git \
    tini \
    > /dev/null && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    wget --no-hsts --quiet https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_NAME}-${MINIFORGE_VERSION}-Linux-$(uname -m).sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    conda clean --tarballs --index-cache --packages --yes && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    conda clean --force-pkgs-dirs --all --yes  && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc

ENTRYPOINT ["tini", "--"]
CMD [ "/bin/bash" ]

# Install and Setup Isaac Lab 2.0.0
# Ref: https://isaac-sim.github.io/IsaacLab/source/setup/installation/binaries_installation.html
RUN apt-get update \
    && apt-get install -y \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab \
    && cd ~/IsaacLab \
    && git checkout v2.0.0
WORKDIR /root/IsaacLab
# Apply patch for rsl-rl package name change
# Ref: https://github.com/isaac-sim/IsaacLab/pull/1808
RUN sed -i 's/rsl-rl@/rsl-rl-lib@/g' ./source/isaaclab_rl/setup.py
RUN ln -s /isaac-sim _isaac_sim
RUN ./isaaclab.sh --conda
RUN . /opt/conda/etc/profile.d/conda.sh \
    && conda activate env_isaaclab \
    && ./isaaclab.sh --install

RUN echo "conda activate env_isaaclab" >> ~/.bashrc

# Install ffmpeg for video recording
# Ref: https://isaac-sim.github.io/IsaacLab/source/how-to/wrap_rl_env.html#wrapper-for-recording-videos
RUN apt-get update \
    && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# # Tools and scripts for Omniverse Farm
# COPY thirdparty/omnicli /omnicli
# COPY scripts/docker/run.sh /run.sh

# # Prevent users from accidentally removing the executable permission.
# RUN chmod +x /omnicli/omnicli && chmod +x /run.sh
# # Prevent users from accidentally saving script files with Windows line endings.
# RUN sed -i 's/\r$//' /run.sh

# Install ROS2 Humble

ENV UBUNTU_CODENAME=focal
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt-get update && apt-get install -y \
    ros-humble-desktop

# Install colcon and other tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-argcomplete \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get -qq clean

# Install additional Python packages
COPY ./requirements.txt /root/requirements.txt
WORKDIR /root
RUN /opt/conda/envs/env_isaaclab/bin/python -m pip install ir-sim
# RUN /usr/bin/python3 -m pip install -r requirements.txt

# Install rosdep
RUN rosdep init
RUN rosdep update

# Install rqt plugins
RUN apt-get install -y ros-humble-rqt-*

# Install ipopt
RUN apt-get update && apt-get install -y \
    coinor-libipopt-dev \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libmetis-dev

# Install ros2 packages
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-pcl-* \
    ros-${ROS_DISTRO}-geometry2 \
    ros-${ROS_DISTRO}-teleop-twist-keyboard \
    ros-${ROS_DISTRO}-navigation2 \
    ros-${ROS_DISTRO}-nav2-bringup \
    ros-${ROS_DISTRO}-joy \
    ros-${ROS_DISTRO}-rqt-* \
    ros-${ROS_DISTRO}-rviz-* \
    ros-${ROS_DISTRO}-diagnostic-updater \
    ros-${ROS_DISTRO}-nav2-behavior-tree \
    ros-${ROS_DISTRO}-behaviortree-cpp-v3

# other packages
RUN curl -sSL https://eieiof11.github.io/ppa/ppa.key -o /usr/share/keyrings/ppa.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ppa.gpg] https://eieiof11.github.io/ppa $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ppa.list > /dev/null
RUN echo "yaml https://eieiof11.github.io/ppa/rosdep.yaml" | tee /etc/ros/rosdep/sources.list.d/99-ppa.list > /dev/null
RUN sudo apt update && rosdep update

RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-casadi-ament \
    ros-${ROS_DISTRO}-jsk* \
    ros-${ROS_DISTRO}-twist-switcher \
    ros-${ROS_DISTRO}-extension-node \
    ros-${ROS_DISTRO}-data-logger \
    ros-${ROS_DISTRO}-common-utils

# make ros2 workspace
RUN mkdir -p /home/user/$ROS_WS/src
WORKDIR /home/user/$ROS_WS
RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && colcon build --symlink-install"

# Setup environment
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc
RUN echo "source /home/user/$ROS_WS/install/local_setup.bash" >> ~/.bashrc
RUN echo "source /usr/share/colcon_cd/function/colcon_cd.sh" >> ~/.bashrc
RUN echo "export _colcon_cd_root=/home/user/$ROS_WS" >> ~/.bashrc
RUN echo "alias run_isaaclab='. ~/IsaacLab/isaaclab.sh'" >> ~/.bashrc
RUN echo "alias isaac_python='. ~/IsaacLab/_isaac_sim/python.sh'" >> ~/.bashrc

# git safe directory
COPY ./ /root/tdmpc_playground
WORKDIR /root/tdmpc_playground
RUN git config --global --add safe.directory /root/tdmpc_playground
# Setup entrypoint
COPY ./ros_entrypoint.sh /
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
