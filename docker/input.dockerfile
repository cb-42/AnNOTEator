FROM python:3.9-slim-buster

WORKDIR /code

COPY . .

RUN pip3 install -r requirements_inference.txt
