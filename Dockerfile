FROM jupyter/scipy-notebook

USER root
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends cmake zlib1g-dev libjpeg-dev xvfb xorg-dev python-opengl libboost-all-dev libsdl2-dev swig && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER $NB_USER
RUN pip install gym[all]
