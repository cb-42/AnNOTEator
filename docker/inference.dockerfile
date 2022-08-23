FROM python:3.9-slim-buster

WORKDIR /code

COPY /inference_src .

RUN pip3 install -r inference_requirements.txt

RUN ./MuseScore-3.6.2.548021370-x86_64.AppImage install --appimage-extract

RUN rm ./MuseScore-3.6.2.548021370-x86_64.AppImage

EXPOSE 8080
