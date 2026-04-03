FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

ARG USERNAME=karasu
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update && apt-get install -y \
    git curl wget vim build-essential pkg-config \
    ffmpeg libsm6 libxext6 sudo \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

RUN chown -R ${USER_UID}:${USER_GID} /workspace

USER ${USERNAME}

CMD ["/bin/bash"]