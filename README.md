# TCRpair README

## 1. Setup

TCRpair uses Tensorflow 2.0 and pandas/numpy for data processing. For a full list of all used libraries, see tcrpair.yml.
We recommend creating a TCRpair specific conda environment (https://anaconda.org/) using tcrpair.yml.
~~~
conda env create -f tcrpair.yml
~~~
Activate the TCRpair conda environment before you run TCRpair
~~~
conda activate tcrpair
~~~

## 2. Usage

For predictions with TCRpair, call tcrpair_predict.py including the required arguments.
To see, which arguments are available including a help text describing the expected value, call
~~~
python tcrpair_predicty.py -h
~~~

