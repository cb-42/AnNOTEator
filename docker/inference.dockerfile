FROM python:3.9-slim-buster

WORKDIR /code

COPY /inference_src .

RUN pip3 install -r inference_requirements.txt

EXPOSE 8080

ENTRYPOINT [ "python", "inference.py" ]
