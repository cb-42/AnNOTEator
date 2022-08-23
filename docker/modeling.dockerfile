FROM python:3.9-slim-buster

WORKDIR /code

COPY /modeling_src .

RUN pip3 install -r modeling_requirements.txt

EXPOSE 8080
