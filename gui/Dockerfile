FROM node:16.0.0-buster-slim as backdoorpony-frontend
RUN npm i -g @vue/cli
WORKDIR /vue
COPY ./gui .
ENTRYPOINT npm install && npm run serve
