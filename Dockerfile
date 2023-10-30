FROM python:3.9.9
RUN apt-get -y update

# Args
ARG user=[username]
ARG userid=[userid]

# opencv-python essential (not installed in Docker)
RUN apt-get -y install libgl1-mesa-glx

# install mothi
RUN pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple mothi

# set workdir to /home/user and copy local directory
WORKDIR /home/${user}
COPY . .

# install QuPath 0.3.2 and set the enviroment variable
RUN python -m paquo get_qupath --install-path ./ 0.4.4
ENV PAQUO_QUPATH_DIR=/home/${user}/QuPath-0.4.4

# set local user... otherwise you can not acces the QuPath project outside of Docker
RUN useradd -u ${userid} ${user}
USER ${user}