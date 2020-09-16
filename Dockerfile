# The first stage `builder` constructs an `fenics` environment to perform the
# required preprocessing. The outcome of the preprocessing is stored in the
# second stage to prevent preprocessing on repeated calls to the container.
FROM quay.io/fenicsproject/stable:latest AS builder

WORKDIR /app 

# install python requirements 
# strip `eventmodule`: not required for preprocessing
COPY requirements.txt ./
RUN pip3 install --upgrade pip
RUN cat requirements.txt | sed '/eventmodule/d' | pip install --no-cache-dir -r /dev/stdin

# extract the brain mesh
ADD brain_meshes.tar.xz ./
COPY perfusion ./perfusion
COPY . .

# preprocessing
RUN cd perfusion && python3 permeability_initialiser.py

FROM quay.io/fenicsproject/stable:latest
WORKDIR /app

# install python requirements 
COPY eventmodule ./eventmodule
COPY requirements.txt ./
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# copy preprocessing results 
COPY --from=builder /app .

ENTRYPOINT ["python3", "API.py"]
