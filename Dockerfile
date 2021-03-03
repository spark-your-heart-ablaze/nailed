FROM ubuntu:16.04

MAINTAINER Viktor Ivanov 'vantek1266@gmail.com'
FROM python:3.7

RUN apt-get update -y
RUN apt-get install -y python-pip python-dev libsm6 libxrender1 libfontconfig1 libice6
RUN pip install --upgrade pip

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]