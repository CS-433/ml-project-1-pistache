# Wait... Is this Higgs Boson?

This repository contains implementations for several binary classification
algorithms as well as a classification pipeline with the goal of
discovering Higgs boson.

### Dataset

The dataset is provided by CERN, and includes a train.csv and a test.csv file.
Here is a [link](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/dataset_files?unique_download_uri=172473&challenge_id=66) to download data.
Here is a [link](https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf) to data description sheet.

### The implemented algorithms are:

- Linear regression using gradient descent
- Linear regression using stochastic gradient descent
- Least squares regression using normal equations
- Ridge regression using normal equations
- Logistic regression using gradient descent or SGD
- Regularized logistic regression using gradient descent or SGD

### How to use this repository

#### Reposity Map
- `helpers.py`: This files includes helper functions and utilities functions.
- `implementations.py`: This file includes the implementations 
of the classification algorithms
mentioned above.
- `run.ipynb`: This notebook contains the implementation of the whole 
classification pipeline using the best method (Ridge Regression) and generates the final predictions
on the test dataset.

#### Necessary Packages
You need to install `numpy` and `matplotlib` libraries before using this repository.

#### Running the classification
In order to run perform classification and generate submission.csv
file, you need to open `run.ipynb` in a jupyter server in
a browser, Visual Studio Code, Pycharm and run all the cells.


### Results
Using our best method, Ridge Regression Classifer, on the test set, we obtained:
- Accuracy: `0.835`
- F1-score: `0.746`
On the AICrowd platform.

### Authors
Amin Asadi Sarijalou ([amin.asadisarijalou@epfl.ch](amin.asadisarijalou@epfl.ch)),
Ahmad Rahimi ([ahmad.rahimi@epfl.ch](ahmad.rahimi@epfl.ch)),
Parham Pourmohamaddi ([parham.pourmohammadi@epfl.ch](parham.pourmohammadi@epfl.ch])).