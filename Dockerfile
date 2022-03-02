FROM continuumio/miniconda3

WORKDIR /app

COPY . ./
# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx

# Create the environment:
RUN conda env create -f environment.yml

# Initialize conda in bash config fiiles:
RUN conda init bash

# Activate the environment, and make sure it's activated:
RUN echo "conda activate perfusion" > ~/.bashrc

# ensure the installation is working and pip is available
RUN python3.9 -m pip install pip --user
RUN python3.9 -m pip install --upgrade pip distlib wheel setuptools cython

RUN cd /app/ && python3.9 -m pip install --no-cache-dir ./in-silico-trial
RUN cd /app/ && python3.9 -m pip install --no-cache-dir -r requirements.txt
RUN export DIJITSO_CACHE_DIR=/patient/.cache

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "perfusion", "python3.9", "API.py"]

