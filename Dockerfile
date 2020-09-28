FROM ubuntu:18.04

RUN apt update
RUN apt install -y --no-install-recommends software-properties-common
RUN add-apt-repository ppa:fenics-packages/fenics
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y fenics python3-h5py python3-lxml python3-coverage python3-vtk7 python3-tables g++

RUN apt install -y ssh

RUN apt install python3-pip --yes
RUN pip3 install --upgrade pip
RUN pip3 install untangle
RUN pip3 install scipy
RUN pip3 install argparse
RUN pip3 install PyYAML

COPY ./brain_meshes /brain_meshes
COPY ./perfusion /perfusion
COPY ./runner.py /runner.py

CMD ["python", "./runner.py"]
