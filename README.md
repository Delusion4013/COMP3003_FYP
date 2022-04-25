## Environment preparation
The entire project runs using Python 3.8.2. A conda environment is configured with important packages as follows:
- tensorflow --version=2.4.1
- numpy --version=1.19.2
- pandas --version=1.4.1
- tensorlflow-addons --version=0.16.1

Commands to create new conda environment:
```shell
conda create -n [env_name] python=[python_version]
```

Commands to install packages:
```shell
conda install [package_name]=[version]
```
If not success, please refer to the official website of different libraries.

# General Introduction

The majority of work is done on nottingham augusta high performance computing, therefore the model script may not run locally. The entire dataset is also stored on the HPC in `home/scych2/data` due to the limitations of local disk storage. 

To use HPC, please refer to the [official website](https://uniofnottm.sharepoint.com/sites/DigitalResearch/SitePages/Access-Compute.aspx).


## Generating MIF data
The generation uses [Open3dQSAR software](http://open3dqsar.sourceforge.net/) to calculate the Molecular Interaction field data. The script used to generate the data could be found in [this repo](scripts/generation.txt).

## File structure
In `scripts/` folder, some useful scripts are included. `transform.py` would take the input MIF data and transformed it into `.npy` format. `predict.py` would load the pre-stored MIF data with train-test split, the model specified and outputs the $R^2$ of the model. `sampleJob.sh` is used on HPC to submit the job to slurm scheduler.

In `models/` folder, different architectures are implemented. The `Model.ipynb` is used as an interactive testing environment.

In `data/` folder, the labeled molecule and binding affinity data are stored in `bf.csv`. The `2yel/` folder contains example output files generated using the scirpt described in [this section](#generating-mif-data). The `b2yel/` folder contains the transformed data using `transform.py`.
