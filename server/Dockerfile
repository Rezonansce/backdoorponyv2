FROM python:3.8.9-slim-buster as backdoorpony-backend
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y wkhtmltopdf
WORKDIR /src
COPY ./server/src/ .
COPY ./server/assets ./assets
COPY ./server/data ./data
ENV FLASK_ENV=development
ENV DEBUGGER=True
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
RUN pip3 install -r requirements.txt --ignore-installed
RUN python -m nltk.downloader stopwords
WORKDIR /src/backdoorpony
CMD [ "python3", "-m", "flask", "run", "--host=0.0.0.0"]
