# Bilevel Denoising Parameter Learning

This is the accompanying code for the PhD Thesis [X]. 

## Installation

```bash
$ conda create --name <env> --file requirements.txt 
$ conda activate <env>
$ pip install pyproximal
```

## Usage

Execute the optimal parameter learning algorithm for the scalar case

```zsh
$ python bdpl.py --dataset <dataset_dir> --scalar --learn <data/reg>
```

Execute the optimal parameter learning algorithm for the patch-dependent case

```zsh
$ python bdpl.py --dataset <dataset_dir> --patch <num_patches> --learn <data/reg>
```