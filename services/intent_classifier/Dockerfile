# syntax=docker/dockerfile:experimental

FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

WORKDIR /src

# setting env
ARG SERVICE_PORT
ENV SERVICE_PORT ${SERVICE_PORT}
ENV PRETRAINED_MODEL_NAME_OR_PATH /src/DNNC-few-shot-intent/roberta_nli/

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

# run service
COPY . /src

HEALTHCHECK --interval=5s --timeout=90s --retries=3 CMD curl --fail 127.0.0.1:1234/healthcheck || exit 1

CMD gunicorn --workers=1 server:app -b 0.0.0.0:1234 --timeout=300

EXPOSE 1234


# GOAL 2: run that service
# FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

# WORKDIR /src

# ARG PRETRAINED_MODEL_NAME_OR_PATH
# ENV PRETRAINED_MODEL_NAME_OR_PATH ${PRETRAINED_MODEL_NAME_OR_PATH}
# ARG SERVICE_PORT
# ENV SERVICE_PORT ${SERVICE_PORT}

# COPY ./requirements.txt /src/requirements.txt
# RUN pip install -r /src/requirements.txt

# RUN python -c 'from transformers import BertTokenizer; BertTokenizer.from_pretrained("bert-base-uncased");'
# RUN python -c 'from transformers import BertForMaskedLM; BertForMaskedLM.from_pretrained("bert-base-uncased");'

# COPY . /src

# HEALTHCHECK --interval=5s --timeout=90s --retries=3 CMD curl --fail 127.0.0.1:${SERVICE_PORT}/healthcheck || exit 1


# CMD gunicorn --workers=1 server:app -b 0.0.0.0:${SERVICE_PORT} --timeout=300
