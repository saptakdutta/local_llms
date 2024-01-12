# Local LLMS
This repository is meant to act as a guide to download and inference with local LLMs. The three inferencing examples are:
- simple question and response
- question and response with RAG
- question and response with langchain database agents

The models used in this repository are:
- Mistral-7B-OpenOrca
- Mistral-7B-Instruct
- Solar-10.7B-SLERP
- Solar-10.7B-Instruct
- Nous-Capybara-34B
- Nous-Hermes-2-Yi-34B
- Mixtral-8x7B-Instruct

All models in this repo are run using the GPTQ format. 

# Getting started
In order to set up this project you will need the repository, and a virtual environment that contains all the required software dependencies.

## Installing GIT
Start by installing `GIT` on your system, which will allow us to clone the repository:
### Linux
Using apt (debian based): 
> sudo apt install git-all

Using dnf (RHEL based):

> sudo dnf install git-all

### MacOS
Use the homebrew package manager
> brew install git

### Windows
> Follow [this tutorial](https://git-scm.com/download/win) to set it up locally

Once git is installed, `cd` into the directory that you want this project to be located in and then clone this repository like so:

> git clone https://github.com/saptakdutta/audio_feature_extraction

You'll be prompted to enter in your gitlab username and password to clone the repo locally.
Now go ahead to the next part to set up the virtual environment

## Setting up the virtual environment
The provided `environment.yml` file contains the dependencies you will require to set up the project. Install conda/miniconda using the [following instructions.](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)

Ensure that your conda install is upto date using:

> conda update conda

Use your python package manager (conda/miniconda/mamba) to cd into the root directory and run the following command:

> conda env create -f environment.yml

Once the new environment has been created, install the other dependencies like so:

> pip3 install --upgrade "git+https://github.com/huggingface/transformers" optimum

> pip3 install --upgrade auto-gptq

> pip3 install --upgrade langchain langchain-community langchainhub gpt4all chromadb

You should now be ready to get up and running

# Huggingface model locations
By default on a debian based system, the models from huggingface hub will be installed to: 

> Home/.cache/huggingface/hub