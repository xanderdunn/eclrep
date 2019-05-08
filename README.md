# Deep Peptide Representation Learning

We present an interface for training, inferencing representations, generative modelling aka "babbling", and data management. All three architectures (64, 256, and 1900 units) are provided along with the trained architectures, the random initializations used to begin evotuning (to ensure reproducibility) and the evotuned parameters. 

For training/finetuning: note that backpropagation of an mLSTM of this size is very memory intensive, and the primary determinant of memory use is the max length of the input sequence rather than the batch size. We have finetuned on GFP-like fluorescent proteins (~120-280aa) on a p3.2xlarge instance (aws) with 16G GPU memory successfully. Higher memory hardware should accommodate larger sequences, as will using one of the smaller pre-trained models (64 or 256).

## Setup

- System requirements: NVIDIA CUDA 8.0 (V8.0.61), NVIDIA cuDNN 6.0.21, NVIDIA GPU Driver 410.79 (though == 361.93 or >= 375.51 should work. Untested), nvidia-docker. The 64-unit model should be OK to run on any machine. The full-sized model will require a machine with more than 16GB of GPU RAM.
- Build docker: `docker build -f docker/Dockerfile.gpu -t eclrep-gpu .` This step pulls the Tensorflow 1.3 GPU image and installs a few required python packages. Note that Tensorflow pulls from Ubuntu 16.04.
- Run docker: `docker/run_gpu_docker.sh`. This will launch Jupyter. Copy and paste the provided URL into your browser. Note that if you are running this code on a remote machine you will need to set up port forwarding between your local machine and your remote machine. See this [example](https://coderwall.com/p/ohk6cg/remote-access-to-ipython-notebooks-via-ssh) (note that in our case jupyter is serving port 8888, not 8889 as the example describes).
- Open up the `unirep_tutorial.ipynb` notebook and get started.

## Data Sources
- Stability data is from David Baker's Science paper ["Global analysis of protein folding using massively parallel design, synthesis, and testing"](https://science.sciencemag.org/content/suppl/2017/07/12/357.6347.168.DC1)

## Weight files

The `unirep_tutorial.ipynb` notebook downloads the needed weight files for the 64-unit and 1900-unit UniRep models.

Weight directories are in data/

- `1900_weights/`: trained weights for the 1900-unit (full) UniRep model
- `256_weights/`: trained weights for the 256-unit UniRep model
- `64_weights/`: trained weights for the 64-unit UniRep model
- `1900_weights_random/`: random weights that were used to initialize the 1900-unit (full) UniRep model for Random Evotuned.
- `256_weights_random/`: random weights that could be used to initialize the 256-unit UniRep model (e.g. for evotuning).
- `64_weights_random/`: random weights that could be used to initialize the 64-unit UniRep model (e.g. for evotuning).
- `evotuned/unirep/`: the weights, as a tensorflow checkpoint file, after 13k unsupervised weight updates on fluorescent protein homologs obtained with JackHMMer of the globally pre-trained UniRep (1900-unit model).
- `evotuned/random_init/`: the weights, as a tensorflow checkpoint file, after 13k unsupervised weight updates on fluorescent protein homologs obtained with JackHMMer of a randomly initialized UniRep (initialized with 1900_weights_random) that was not pre-trained at all (1900-unit model).


## Description of files in this repository
- unirep_tutorial.ipynb - Start here for examples on loading the model, preparing data, training, and running inference. 
- unirep_tutorial.py - A pure python script version of the tutorial notebook.
- unirep.py - Interface for most use cases.
- custom_models.py -  Custom implementations of GRU, LSTM and mLSTM cells as used in representation training on UniRef50
- data_utils.py - Convenience functions for data management.
- formatted.txt and seqs.txt - Tutorial files.
