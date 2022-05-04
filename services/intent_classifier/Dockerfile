# syntax=docker/dockerfile:experimental

FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime as base

WORKDIR /src

# setting env
ARG SERVICE_PORT
ENV SERVICE_PORT ${SERVICE_PORT}
ARG TRAIN_DATA_PATH 
ENV TRAIN_DATA_PATH ${SERVICE_PORT}

COPY . /src

# install reqs
RUN apt-get update \
    && apt-get -y install git \
    && apt-get -y install wget \
    && apt-get -y install unzip

COPY ./requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt

# Downloading model
COPY ./setup.sh /src/setup.sh
RUN  ["sh", "/src/setup.sh"]

RUN python -c 'from transformers import RobertaTokenizer; RobertaTokenizer.from_pretrained("roberta-base");'
RUN python -c 'from transformers import RobertaModel; RobertaModel.from_pretrained("roberta-base");'

FROM base as test

ENV TRAIN_DATA_PATH ./data/example.json
COPY ${TRAIN_DATA_PATH} /src/train_data.json

RUN  ["sh", "/src/test.sh"]

FROM base as deploy

COPY ${TRAIN_DATA_PATH} /src/train_data.json

# run service
HEALTHCHECK --interval=5s --timeout=90s --retries=3 CMD curl --fail 127.0.0.1:${SERVICE_PORT}/healthcheck || exit 1

CMD gunicorn --workers=1 server:app -b 0.0.0.0:${SERVICE_PORT} --timeout=300

EXPOSE ${SERVICE_PORT}
