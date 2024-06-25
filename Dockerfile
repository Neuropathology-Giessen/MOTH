FROM python:3.9.9 as build
# Args
ARG USERNAME=moth
ARG QUPATH_VERSION=0.4.4

# paquo and opencv essential
RUN apt-get -y update && apt-get -y install libgl1 \
    && apt-get -y install libxtst6

# create user
RUN groupadd -r ${USERNAME} && useradd -r -g ${USERNAME} ${USERNAME}
WORKDIR /home/${USERNAME}

# copy local directory
COPY . .

# install local mothi version
RUN pip install .

# install QuPath 0.4.4 and set the enviroment variable
RUN python -m paquo get_qupath --install-path ./ ${QUPATH_VERSION}
ENV PAQUO_QUPATH_DIR=/home/${USERNAME}/QuPath-${QUPATH_VERSION}


FROM build as devbuild
# install sphinx dependencies to build the documentation local
RUN apt-get -y install python3-sphinx
RUN pip install .[docs]

# dev installation for paper and workflow related work 
RUN pip install opencv-python
RUN pip install matplotlib
RUN pip install progress



FROM devbuild as dev
USER ${user}


FROM devbuild as workflow
RUN pip install torch
RUN pip install torchvision
USER ${user}


FROM build as prod
USER ${user}