#!/bin/bash

module load StdEnv/2020  gcc/9.3.0 root/6.26.06 python/3.9

# Install the package in editable mode
pip install -e .

# Export the specified path to the PATH environment variable
export PATH=$PATH:/project/def-mdanning/rhall02/gnn-plots/.local/bin
#export PATH=$PATH:/home/private/EmergingJetTagger/gnn-plots/
