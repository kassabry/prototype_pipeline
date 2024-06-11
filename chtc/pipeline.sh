#! /bin/bash

##https://superuser.com/questions/679580/how-do-i-cd-in-to-the-directory-made-by-tar -- check if this actually works in error log
#gzip -d active-learning-drug-discovery.tar.gz | tar -xzf active-learning-drug-discovery.tar

tar -xzf pipeline.tar.gz

rm pipeline.tar.gz #--- remove the tar file from the server

cd pipeline

export HOME=$PWD
wget -q https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh -O miniconda.sh
sh miniconda.sh -b -p $HOME/miniconda3
rm miniconda.sh
export PATH=$HOME/miniconda3/bin:$PATH

source $HOME/miniconda3/etc/profile.d/conda.sh
hash -r
conda config --set always_yes yes --set changeps1 no

# Install packages specified in the environment file
conda env create -f environment.yml

# Activate the environment and log all packages that were installed
conda activate chemvae
conda list

#pip install -e .

# Balanced Forest
python pipeline.py --data_location=pria_data --sample_size=15000 --number_of_seeds=3 --budget=96 --number_of_iterations=25 --model="forest_balanced"

# Unweighted Forest
#python pipeline.py --data_location=pria_data --sample_size=15000 --number_of_seeds=3 --budget=96 --number_of_iterations=25 --model="forest_none"

# MLP
#python pipeline.py --data_location=pria_data --sample_size=15000 --number_of_seeds=3 --budget=96 --number_of_iterations=25 --model="mlp"


conda deactivate