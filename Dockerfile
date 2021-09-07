FROM python:3.9

WORKDIR /home

COPY code/scripts code
COPY data data
COPY requirements.txt ./

RUN apt-get update
RUN apt-get install -y p7zip-full
RUN pip install --user -r requirements.txt
WORKDIR code/
