FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime as backdoorpony-backend 
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y wkhtmltopdf
# RUN apt-get install --upgrade --reinstall python3.10 -y
RUN apt-get install python3-pip -y
WORKDIR /src
COPY ./server/src/ .
COPY ./server/assets ./assets
COPY ./server/data ./data
ENV FLASK_ENV=development
ENV DEBUGGER=True
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
RUN pip3 install -r requirements.txt --ignore-installed
RUN pip3 install --upgrade --force-reinstall torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN python3 -m nltk.downloader stopwords
WORKDIR /src/backdoorpony
CMD [ "python3", "-m", "flask", "run", "--host=0.0.0.0"]
