# Replication package for the paper "Analyzing techniques for duplicate detection on Q&A websites for game development"

This repository contains all of the code used for extracting and processing data from the Stack Exchange data dump, and training and evaluating the duplicate question detection techniques presented in the paper.

## The data

**The data we used in our study is available on [Zenodo](https://zenodo.org)**. This data includes everything we used in our study, including all of the results, measures, and the models trained throughout the methodology. More information about how the data is organized can be found in [the README file in the data/ directory](data/README.md).

In this repository we provide mock datasets that simulate the ones used in our study. These datasets contain a small set of questions and duplicate pairs that can be used to test the code we provide.

### Benchmark datasets

We provide three datasets that can be used for evaluating other duplicate detection methodologies and comparing with our results. **These datasets have been packaged separately and can also be found on [Zenodo](https://zenodo.org).**

There are two datasets focused on game development based on questions from [Stack Overflow](https://www.stackoverflow.com) and [the Game Development Stack Exchange](https://gamedev.stackexchange.com). The third dataset is comprised of five randomly selected samples of equal size collected from Stack Overflow. All of these datasets were extracted from the June 2021 [Stack Exchange Data Dump](https://archive.org/details/stackexchange). More information about the datasets is available in our paper.

## Using the code

All of the code used in our paper can be found in the [code/ directory](code/). **The scripts are written in Python 3.9** and use several libraries that need to be installed for their correct execution. The code also uses features that are exclusive to Unix systems, and executing them on other environments may lead to errors. More information about the code can be found in [the README file in the code/ directory](code/README.md).

To help executing and reproducing our results, we have included a [requirements.txt file](requirements.txt) that can be used to install the required packages. We also provide a [Dockerfile](Dockerfile) that can be used to create and run a container with all the requirements for running the code. Please read the following sections for instructions on how to set up a virtual environment or Docker container for running the code.

We do note that there are still hardware limitations for running the code. We recommend using a system with at least 32GB of RAM and 200Gb of storage space. We also recommend using a CUDA-capable GPU to reduce the time it takes to calculate the embeddings based on deep-learning techniques. In case you do not have access to a CUDA-capable GPU, you can use Google Colab for free to compute these embeddings. In the [notebooks/ directory](code/notebooks) we provide [an example notebook](code/notebooks/make_embeddings_colab_gpu.ipynb) that can be used on Google Colab for computing embeddings.

### Virtual environment

You can use your prefered method for creating a virtual environment on your system. conda and venv are two popular options for this task:

##### Creating a conda environment

Use the following commands while on the root directory to create a conda environment. Note that you need to install [conda](https://docs.conda.io/en/latest/) prior to using these.

1. Create the environment with `conda create -n gamedev_dups python=3.9`
2. Activate the environment with `conda activate gamedev_dups`
3. Install the required packages with `conda install -c conda-forge --file requirements.txt`

##### Creating an environment with venv

Use the following commands  while on the root directory to create an environment using venv. Note that you need to have Python 3.9 installed to use these.

1. Create the environment with `python3 -m venv gamedev_dups`
2. Activate the environment with `source gamedev_dups/bin/activate` on Unix systems or `gamedev_dups\Scripts\activate` on Windows.
3. Install the required packages with `pip install -r requirements.txt`

### Docker container

Please follow these steps to create a Docker container to run the code:

1. Install [Docker](https://www.docker.com/) on your system if you have not already done so;
2. On a terminal or command prompt, type the following command to create a Docker image based on the repository: `docker build -t dup_questions .`
3. After Docker has finished building the image, use the following command to launch a container based on the image and use it in interactive mode: `docker run -dit dup_questions /bin/bash`
4. After executing the scripts, log out of the Docker container using CTRL+D and use the following command to copy the data from the container back to your system: `docker cp dup_questions:~/data .`

#### **BEFORE EXECUTING THE CODE**

We highly recommend that you perform a test run of the code using the mock data provided in this repository before attempting to use other data. To do this, just type `python3 full_pipeline.py` while in the `code/` directory on your terminal or Docker container. This should take a few minutes and should complete with no errors.

After performing a successful test run, **please change the values in the [consts.py](code/scripts/utils/consts.py) file to your prefered ones.** We have changed some portions of the file to make the test runs quicker. If you wish to use the same parameters as we did in our study, replace the [consts.py file](code/scripts/utils/consts.py) with [the one that shows the values used in our study](code/scripts/utils/consts_used_in_the_paper.py).

#### Executing the code from scracth 

Please follow the following steps to execute the code from scratch, i.e., starting from the data from the [Stack Exchange Data Dump](https://archive.org/details/stackexchange).

1. Download the Posts.xml and PostLinks.xml files for Stack Overflow from the data dump;
2. Replace the existing files in the [data/stackoverflow/raw/ directory](data/stackoverflow/raw) with the ones that you just downloaded. Do not alter the names of the files (keep them as Posts.xml and PostLinks.xml);
3. Download the archive for the Game Development Stack Exchange from the data dump;
4. Replace the existing archive in the [data/gamedev\_se/raw/ directory](data/gamedev_se/raw/) with the one that you just downloaded;
5. Execute the command `python3 full_pipeline.py` while in the `code/` directory to run the whole pipeline from start to finish, or execute the scripts following the order described in [the README file in the code/ directory](code/README.md).

#### Executing the code using our data

To execute the code using our data, download the data package available on [Zenodo](https://zenodo.org) and unzip it on the [data/ directory](data/). You can then execute the script or notebooks in the `code/` directory in any order you like.
