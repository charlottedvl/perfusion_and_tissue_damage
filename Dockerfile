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

ENTRYPOINT ["python3.9", "API.py"]
