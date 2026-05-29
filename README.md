# Deep_repro

Industrial tools used to train and deploy machine learning models are valued for their reliability, flexibility, and ease of use. Drawing from our experience working with large, constantly evolving codebases (particularly in deep learning research projects) we would like to share a simple yet effective setup for making experiments both traceable and reproducible.

This repo is simple implementation of the tools largely used for reproducibility and tracking in deep learning. A more detailed example using CNN and an MLP is available in the [Deep_Learning_reproducibility](https://github.com/J-ally/Deep_repro) repository.

## Structure of the repo

The important files of the repo are the following :
- `pyproject.toml`, `uv.lock` and `.python-version`: UV configuration files. It is used by uv to create the virtual environment and install the dependencies.
- `train.py`: the main script to run the training of the model, it contains the main function and the training loop.
- `dataops/pretraitment.py`: the script to download and preprocess the dataset.
- `conf/config.yaml`: the configuration file for the experiment, it contains all the hyperparameters and settings for the experiment.

# Installing everything

Once uv is installed ([uv install doc](https://docs.astral.sh/uv/getting-started/installation/)), to install the required dependencies, you can use the following command:

```bash
uv sync
```

Then you need to download the dataset :
```bash
uv run dataops/pretraitment.py
```

# Running the code
To run the code, you can use the following command:

```bash
uv run train.py
```

# Inspecting the results

Once the python is loaded correctly in your shell (if not automatically done, you can activate it with $ source .venv/bin/activate at the root of the repo), you can launch MLflow. To inspect the results, you can use the following command:

```bash
mlflow ui
```

If there is already a process using the port you can stop it using this command :

```bash
lsof -ti :5000 | xargs kill -9
```

## A note on the dataset label description

Each example is assigned to one of the following labels:
- 0 T-shirt/top
- 1 Trouser
- 2 Pullover
- 3 Dress
- 4 Coat
- 5 Sandal
- 6 Shirt
- 7 Sneaker
- 8 Bag
- 9 Ankle boot
