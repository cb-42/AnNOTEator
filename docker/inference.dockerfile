FROM python:3.9-slim-buster

WORKDIR /code

COPY /src .

RUN pip3 install -r inference_requirements.txt
