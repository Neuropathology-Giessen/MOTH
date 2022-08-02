FROM python:3.9.9

RUN apt-get -y update
RUN apt-get -y install openslide-tools
RUN apt-get -y install python3-openslide
RUN apt-get -y install libgl1-mesa-glx
RUN apt-get -y install git
RUN apt-get -y install build-essential 

RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install openslide-python
RUN pip install opencv-python
RUN pip install paquo
RUN pip install shapely
RUN pip install tifffile
RUN pip install torch
RUN pip install torchvision
RUN pip install progress

WORKDIR /home/tkauer/
COPY . .
RUN pip install -e .

RUN useradd -u 1006 tkauer