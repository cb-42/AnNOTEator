FROM python:3.9-slim-buster

WORKDIR /code

COPY /src .
COPY /pretrained_models/demucs ./pretrained_models/demucs

RUN pip3 install -r input_requirements.txt
