# icml2018-real-datasets

Code used to run tests using real datasets.

## How to run an experiment

In order to run an experiment, first prepare an `experiment_config.json` file (use `experiment_config_example.json` as an example, removing comments). Then run

```python3 run_experiment.py /path/to/experiment_config.json```.

The experiment results will appear on the output_folder set in the experiment_config file.

## How to add a new dataset

In order to add a new dataset, create a folder inside `./datasets/` with the name of the dataset. Inside, create a `config.json` file containing the dataset configurations and a `data.csv` containing the dataset samples (including a header line).

## Pre-requisites:
- Python 3
- Numpy + Scikit-learn
- LAPACK + OpenBLAS (both optional but recommended)
