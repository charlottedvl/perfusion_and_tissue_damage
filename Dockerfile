FROM quay.io/fenicsproject/stable:latest

WORKDIR /app 
COPY requirements.txt ./
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

ADD brain_meshes.tar.xz ./
COPY perfusion ./perfusion
COPY runner.py runner.py
COPY . .

ENTRYPOINT ["python3","runner.py"]
