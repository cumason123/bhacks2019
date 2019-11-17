FROM ubuntu:16.04

RUN apt-get update && apt-get install -y \
  git \
  python3-pip \
  sudo
RUN pip3 install --upgrade pip
COPY . /bhacks2019
WORKDIR /bhacks2019
RUN pip3 install -r ./requirements.txt
CMD ["python3", "app.py"]

