# BackdoorPony

A testing tool with a web-based GUI for testing neural networks against multiple backdoor attacks.


## Setup
---
### Automatic (with Docker)

Assuming docker and docker-compose are installed and in the path.

Open a terminal window (easiest through right-click in Visual Studio Code on README.md --> Open in Integrated Terminal)

```bash
docker-compose up -d
```

Now Docker will build the images (this might take a few minutes but will only be done on the first run) and run them. After this is done you should be able to see the GUI on http://localhost:8080 and the backend on http://localhost:5000

### Manual

#### Start the server

Assuming Python 3.8 is installed and in the path.

Intial setup.

```bash
cd server
python3.8 -m venv env
source env/bin/activate
cd src
pip install -r requirements.txt --ignore-installed
cd backdoorpony
python -m flask run --host=0.0.0.0
```

Concurrent runs.

```bash
cd server
source env/bin/activate
cd src/backdoorpony
python -m flask run --host=0.0.0.0
```

#### Start the GUI

Assuming Node.JS and NPM are installed and in the path.

Initial setup.

```bash
cd gui
npm install
npm run serve
```

Concurrent runs.

```bash
cd gui
npm run serve
```
---
## Running
When nothing was changed in the docker-compose.yml or one of the Dockerfile files
```
docker-compose up -d
```
is sufficient to start the project.
If something has changed in one of these files run the following commands.
```bash
docker-compose down
docker-compose build
docker-compose up -d
```

---
## Test coverage
To get a coverage report for the tests you must attach a shell to the python container and run the following commands.
This will be automated in the pipeline in the near future.
```
cd server
source env/bin/activate
cd src
coverage run --branch -m unittest discover -p test_*.py
coverage report -m
```
