FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3.7 python3-pip sudo && \
    apt-get install -y python3.7-dev build-essential && \
    apt-get install -y pkg-config libhdf5-100 libhdf5-dev cmake && \
    apt-get clean

RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN ln -s /usr/bin/python3.7 /usr/bin/python

RUN python -m pip install --upgrade pip

ADD requirements.txt .

RUN python -m pip install -r requirements.txt

COPY . .

WORKDIR ./run