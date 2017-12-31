FROM jupyter/tensorflow-notebook:82b978b3ceeb

USER root
RUN apt-get update \
 && apt-get install -y \
        cmake \
        libboost-all-dev \
        libjpeg-dev \
        libsdl2-dev \
        libx11-6 \
        python-opengl \
        swig \
        xorg-dev \
        xvfb \
        zlib1g-dev \
 && rm -rf /var/apt/lists/*

USER $NB_USER
RUN pip3 install \
        Box2D \
        gym[all]
