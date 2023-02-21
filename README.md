# SemiNMF-Autoencoders
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
![build](https://img.shields.io/circleci/project/github/badges/shields/master)

This repository contains the code for the experiments of the semi-NMF based regularization objective over the training of multiple autoencoder architectures for unsupervised hyperspectral unmixing. Design of the semi-NMF objective can be found in `Algorithm.pdf`. The corresponding research paper will be made available on arXiv soon.

## Dependencies
* __Docker 19.03.12__
* __PyTorch 1.9.0__
* __Python 3.7.10__

# Installation

## Data
The dataset is publicly available and can be found [here](https://rslab.ut.ac.ir/data).<br>
Download the Samson dataset from the above-mentioned source. Follow the directory tree given below:<br>
```
|-- [root] HyperspecAE\
    |-- [DIR] data\
        |-- [DIR] Samson\
             |-- [DIR] Data_Matlab\
                 |-- samson_1.mat
             |-- [DIR] GroundTruth
                 |-- end3.mat
                 |-- end3_Abundances.fig
                 |-- end3_Materials.fig
```

## From Docker (Recommended)
Using a docker image requires an NVIDIA GPU.  If you do not have a GPU please follow the directions for [installing from source](#source).
In order to get GPU support you will have to use the [nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) plugin.
The docker image is cached on the GPU with id 0. In case of OOM errors at training, pass two GPUs.
``` bash
# Build the Dockerfile to create a Docker image.
docker build -t dgoel04/snreg:1.0 .

# This will create a container from the image we just created.
docker run -it --gpus '"device=gpu-ids"' dgoel04/snreg:1.0
```

## <a name="source"></a>From source:
1) Install the data by following the steps shown under installation.

2) Clone this repository.  
   `git clone https://github.com/dv-fenix/SemiNMF-Autoencoders.git`  
   `cd SemiNMF-Autoencoders`
   
3) Install the requirements given in `requirements.txt`.  
   `python -m pip install -r requirements.txt`
 
4) Change working directory.  
   `cd run` 

# Run Experiments

## Training the Autoencoders
The code is fairly modular and can be run from the terminal.
``` bash
# For more information on the optional experimental setups and configurations.
python ../src/train.py --help

# You can manually change the arguments in samson_train.sh to choose the different autoencoder configurations.
sh samson_train.sh
```
Please make sure that all the arguments are to your liking before getting started with the training!

## Abundance Map and End-Member Extraction
Please ensure that the arguments contained within `extract.sh` match those used in `samson_train.sh` during training.
``` bash
# You can manually change the arguments in experiments.sh to choose the different configurations.
sh extract.sh
```

# References
The autoencoders used in our experiments were first suggested in the following work.
```
@article{palsson2018hyperspectral,
  title={Hyperspectral unmixing using a neural network autoencoder},
  author={Palsson, Burkni and Sigurdsson, Jakob and Sveinsson, Johannes R and Ulfarsson, Magnus O},
  journal={IEEE Access},
  volume={6},
  pages={25646--25656},
  year={2018},
  publisher={IEEE}
}
```
The code for the autoencoders was open-sourced earlier and can be found [here](https://github.com/dv-fenix/HyperspecAE).
This repository uses a lot of the same code with changes made according to the required experimental configurations.
