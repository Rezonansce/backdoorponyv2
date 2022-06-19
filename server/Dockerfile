FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime as backdoorpony-backend
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update &&\
    apt-get upgrade -y &&\
    apt-get install python3-pip libgl1 libsndfile1-dev -y
WORKDIR /src
COPY ./server/src/ .
COPY ./server/assets ./assets
COPY ./server/data ./data
ENV FLASK_ENV=development
ENV DEBUGGER=False
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DGLBACKEND=pytorch
RUN pip3 install -r requirements.txt --ignore-installed &&\
    python3 -m nltk.downloader stopwords wordnet omw-1.4 &&\
    pip3 install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html
WORKDIR /src/backdoorpony
CMD [ "python3", "-m", "flask", "run", "--host=0.0.0.0"]
