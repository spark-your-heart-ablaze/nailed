FROM ubuntu:16.04

MAINTAINER Viktor Ivanov 'vantek1266@gmail.com'
FROM python:3.7

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
RUN apt-get install -y python-pip python-dev libsm6 libxrender1 libfontconfig1 libice6 ffmpeg libxext6
RUN pip install --upgrade pip

EXPOSE 4343

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "4343" ]