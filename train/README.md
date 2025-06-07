## Table of Contents

- [1. Workplace](#1-workplace)
- [2. Installation](#2-installation)
- [3. Usage](#3-usage)
  - [3.1 Running the Server](#31-running-the-server)
  - [3.2 Running the Client](#32-running-the-client)


## 1. Workplace

Firstly ensure that python and pip has been installed. 
Ideally set up a virtual environment (conda or pyenv or personal choice)

(For the project that the owner is doing, python=3.8 on Jerson Nano)

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

## 2. Installation
Stay in train/ folder and run the following command

```bash
pip install -r requirements.txt
```

Run the server 

```bash
python3 server/server.py
```

## 3. Usage

Before starting the training run, one must obviously first have the dataset (in yolo structure and format) that is used to train. It is recommended to put it in the data folder, same path with the data.yaml file for ease of use. 
Please read the data.yaml file for instruction on how to put the correct path.

Furthermore, within the task.py of both client and server folder, there are hyperparameters that one can change for an optimal training run (inside the def train() function)

### 3.1 Running the Server

Run 

```bash
python3 server/server.py -h 
```
For information on the training parameters, else, it will default to the value set (descripted in the -h flag)

For default value, run:
```bash
python3 server/server.py 
```

For non-default value, one could run:
```bash
python3 server/server.py --rounds 4 --port 9000 --min-fit-clients 2  
```
### 3.2 Running the Client
Run 

```bash
python3 client/client.py -h 
```
For information on the client's parameters, else, it will default to the value set (descripted in the -h flag)

For now client only has --server-address as parameters

So run:
 ```bash
python3 client/client.py --server-address 34.32.156.4:8080 # for example
```



