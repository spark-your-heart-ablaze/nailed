FROM ubuntu:latest

MAINTAINER Viktor Ivanov 'vantek1266@gmail.com'


RUN apt-get update -y && \
    apt-get install -y python-pip python-dev libsm6 libxrender1 libfontconfig1 libice6

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]