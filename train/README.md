# Federated Learning 

## Working environment
Firstly ensure that python and pip has been installed.
Ideally set up a virtual environment (conda or pyenv or personal choice)

Conda:
```bash
conda create -n my_env python=x #replace x with version of choice and my_env with name name of choice
```
Activate
```bash
conda activate my_env
```

Python env:
```bash
python(x) -m venv /path/to/new/virtual/environment # dont create this within the git repo and replace (x) with wanted version
```
Activate
```bash
source /path/to/new/virtual/environment/bin/activate
```


## Installing the dependency 
Stay in train folder and run the following command

```bash
pip install -r requirements.txt
```

## Run the server 

```bash
python3 server/server.py
```


## Run the client 

```bash
python3 client/client.py
```


