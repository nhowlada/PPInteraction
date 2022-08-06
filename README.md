# Protein Protein Interaction Prediction

### [Video]() | [Paper]()

Repository for the paper [Protein Protein interaction prediction]().

If you have questions or any trouble running some of the code, don't hesitate to open an issue or ask me via nhowlada@uno.edu. I am happy to hear from you!
## Usage

Either train your own model or use the weights I provide in this repository and do only inference.

### 1. Get Protein Embeddings

The design operates on embeddings obtained from single sequences of amino acids without additional evolutionary information as in profile embeddings. Just [download](google drive link) the embeddings and place them in `data_files`.

Alternately, the [bio-embeddings](https://pypi.org/project/bio-embeddings/) library can be used to build embedding files from ``.fasta`` files. Simply modify the location in the necessary configuration file, such as `configs_clean/linear attention.yaml`, to point to your embeddings and remapping file in order to use the embeddings.

### 2. Setup environment

If you are using conda, you can install all dependencies like this.

```
conda create --name PPInteraction python=3.9.7
conda activate PPInteraction
pip3 install -r requirements.txt
```

### 3.1 Training
As mentioned in Step 1, make sure the [embeddings]() are in the files named in `configs_clean/linear attention.yaml`. The next step is to begin your training:

```
python3 train.py --config configs_clean/linear_attention.yaml
tensorboard --logdir=runs --port=6006
```

Navigate your browser to `localhost:6006` to view the model training.

### 3.2 Inference

You can either use your own trained weights stored in the ***runs*** folder or download the [trained model weights]() and place them in the folder ***runs***, then modify the path and folder name in `configs_clean/inference.yml` to utilize them. Inferring can begin with the following command:

```
python3 inference.py --config configs_clean/inference.yaml
```

The results are written to the `inference/inference.txt` file in the `output` directory.

## Architecture

![architecture](https://github.com/nhowlada/PPInteraction/blob/main/generalize_model.png)

## Setup

Python 3.9.7 dependencies:

- Bio==1.3.9
- dgl==0.9.0
- h5py==3.7.0
- importlib_metadata==4.12.0
- joblib==1.1.0
- matplotlib==3.5.2
- numpy==1.23.1
- pandas==1.4.3
- pyaml==21.10.1
- PyYAML==6.0
- scikit_learn==1.1.1
- seaborn==0.11.2
- torch==1.12.0
- torchvision==0.13.0
- tqdm==4.64.0
- tensorboard

You can use `pip3 install requirements.txt` file to install all of the above dependencies.

```
pip3 install -r requirements.txt
```


## Train and Inference with Docker image

### Pull image from Docker registry
```
docker pull nhowlada/ppinteraction:latest
```

> ### (Alternatively)
> ### [Clone the repo form the github](https://github.com/nhowlada/PPInteraction)
> ```
> git clone git@github.com:nhowlada/PPInteraction.git
> ```
> ### [Build with docker](https://hub.docker.com/repository/docker/nhowlada/ppinteraction)
> ```
> docker build -f Dockerfile -t nhowlada/ppinteraction .
> ```


### Run docker container
```
docker run --name ppinteraction --gpus 1 -v /PATH_DATA_FILES/:/app/data_files/ -it nhowlada/ppinteraction:latest
```

### Training with docker
```
source activate PPInteraction
python3 train.py --config configs_clean/linear_attention.yml
```

### Inference with docker
```
source activate PPInteraction
python3 inference.py --config configs_clean/inference.yml
```


### Check **runs** folder for saved model and training log. 
```
cd runs
```
### Copy Training log:
```
docker cp -r ppinteraction:/app/runs/FOLDER_NAME .
```
### Copy Inference log:
```
docker cp ppinteraction:/app/output/inference/inference.txt .
```

