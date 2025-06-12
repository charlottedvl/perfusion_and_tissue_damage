FROM continuumio/miniconda3

WORKDIR /app

COPY . ./
# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx

# Initialize conda in bash config files:
RUN conda init bash
RUN conda update -n base -c defaults conda -y

# Create the environment:
RUN conda create -n perfusion -c conda-forge fenics python=3.9 -y

# Activate the environment, and make sure it's activated:
RUN echo "conda activate perfusion" > ~/.bashrc

# ensure the installation is working and pip is available
RUN python3.9 -m pip install pip --user
RUN python3.9 -m pip install --upgrade pip distlib wheel setuptools cython

RUN cd /app/ && python3.9 -m pip install --no-cache-dir ./in-silico-trial
RUN cd /app/ && python3.9 -m pip install --no-cache-dir -r requirements.txt
RUN export DIJITSO_CACHE_DIR=/patient/.cache

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "perfusion", "python3.9", "API.py"]

