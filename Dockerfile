FROM python:3.7.6-buster

RUN mkdir /usr/app
WORKDIR /usr/app

COPY ./requirements.txt /usr/app/requirements.txt
RUN python3 -m pip install -r requirements.txt
