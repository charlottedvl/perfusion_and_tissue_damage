# The first stage `builder` constructs an `fenics` environment to perform the
# required preprocessing. The outcome of the preprocessing is stored in the
# second stage to prevent preprocessing on repeated calls to the container.
FROM quay.io/fenicsproject/stable:latest AS builder

WORKDIR /app

# install python requirements
COPY requirements.txt ./
RUN pip3 install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# extract the brain mesh
ADD brain_meshes.tar.xz ./
COPY perfusion ./perfusion
COPY oxygen ./oxygen

# preprocessing
RUN cd perfusion && python3 permeability_initialiser.py

FROM quay.io/fenicsproject/stable:latest
WORKDIR /app

# install python requirements
COPY in-silico-trial ./in-silico-trial
COPY requirements.txt ./

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
        python3 \
        python3-pip \
        python3.9 \
        python3.9-distutils

# ensure the installation is working and pip is available
RUN python3.9 -m pip install pip --user
RUN python3.9 -m pip install --upgrade pip distlib wheel setuptools
RUN cd /app/ && python3.9 -m pip install --no-cache-dir ./in-silico-trial

# the other requirements to run
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# copy all local contents
COPY . .

# overwrite with preprocessing results
COPY --from=builder /app .

ENTRYPOINT ["python3.9", "API.py"]
