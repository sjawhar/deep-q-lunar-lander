FROM jupyter/tensorflow-notebook:033056e6d164

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
RUN pip install gym[all]
RUN pip install Box2D
