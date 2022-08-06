FROM python:3.9-slim-buster

WORKDIR /code

COPY /input_src .
COPY /pretrained_models/demucs ./pretrained_models/demucs

RUN pip3 install -r input_requirements.txt

EXPOSE 8080

ENTRYPOINT [ "python", "input.py" ]
