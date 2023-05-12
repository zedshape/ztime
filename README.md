# Z-Time: Efficient and Effective Interpretable Multivariate Time Series Classification

Z-Time is an efficient and effective interpretable multivariate time series classifier. It employs temporal abstractions and builds temporal relations of event intervals to create interpretable features across multiple time series dimensions.

# Dependencies
This application needs the latest version of the following packages and Python >= 3.8. We restrict the usage of the libraries for clear reproducibility of the research. For details, please refer to `requirements.txt`.

`numpy`, `pandas`, `pyts`, `scipy`

Numpy, pandas, and scipy are used for dataset handling. Pyts is used for temporal abstractions.

# How to use

Our helper.py focuses on automating the process of reproducing the experiment. The experiments can be reproducible by mainly using the following functions:

`simpleTrial`: Run one trial of Z-Time classification given parameters.
  - data: variable that includes both training and test sets.
  - alphabet_size: $\lambda_1$ parameter.
  - alphabet_size_slope: $\lambda_2$ parameter.
  - window_size: $w$ parameter.
  - split_no: $\rho$ parameter.
  - step: $\sigma$ parameter.
  - multivariate: indication if the dataset is multivariate. The default is `True`.
  - classifier: `Ridge`, `Elastic`, or `Lasso`. The default is `Ridge`.

`randomizedSearch`: Run a randomized search of $k$ iterations.
  - data: variable that includes both training and test sets.
  - steps: list of $\sigma$ parameters.
  - alphabet_sizes: list of $\lambda_1$ parameters.
  - alphabet_sizes_slope: list of $\lambda_2$ parameters.
  - window_sizes: list of $w$ parameters.
  - split_nos: list of $\rho$ parameters.
  - cycle: the number of iterations for randomized search.
  - multivariate: indication if the dataset is multivariate. The default is `True`.
  - classifier: `Ridge`, `Elastic`, or `Lasso`. The default is `Ridge`.

Please refer to the paper for a detailed explanation of the parameters.

Note that the dataset should follow the format of a 3D numpy array with the shape (size, dimension, length). We have one instance incorporating both training and test sets. The data structure should be as follows to run Z-Time using the helper.py:

- data["TRAIN"]["X"]: Training set in 3D numpy array format with the shape (size, dimension, length)
- data["TRAIN"]["y"]: List of training labels
- data["TEST"]["X"]: Training set in 3D numpy array format with the shape (size, dimension, length)
- data["TEST"]["y"]: List of test labels

The example of `simpleTrial` is available in `example.py`. The example of `randomizedSearch` is available in `example2.py`.

# Regularization

Z-Time can be used together with various regularization techniques depending on different use-cases. However, to obtain interpretability, linear classifiers are recommended. Our code provides some of them as an options that are widely used in the area. The default option is set as `Ridge`, which is used in our paper.

You may put one of the following options when you call `simpleTrial` or `randomizedSearch`. Also, you can use different classifiers in scikit-learn.

- `Ridge`: Ridge (l2) classifier (parameters are chosen following Rocket).
- `Elastic`: Elastic net classifier (parameters are chosen following PETSC).
- `Lasso`: Lasso (l1) classifier (parameters are chosen following PETSC).

# Datasets

## The UCR/UEA time series datasets

We use the multivariate and univariate time series datasets in the UCR/UEA repository. These datasets can be found at [here](http://www.timeseriesclassification.com). Z-Time uses a 3D numpy array form of those datasets with the shape (size, dimension, length). For your reference, we have a converted BasicMotions dataset in our repository. 

## Synthetic datasets

In our experiment, we use three synthetic datasets (SYN1, SYN2, and SYN3), and they are all available in a pickled form in the `data` folder. You can use the `simpleTrial` function in the same way as other datasets. Here are details of our synthetic data.

- SYN1: two sinusoidal concave/convex patterns are imported in the first and last 25% of the time series with random noises. Sinusoidal patterns have a peak size of 2 or -2. Example figures can be found in the paper.
- SYN2: two peaks are imported at random time points of the time series with random noises, which keep the orders in two dimensions, and different order means different class. The dataset has two classes. Noises follow the standard random distribution, and peaks are randomly chosen from 10 and -10. Each class always keeps the same order of peaks. Example figures can be found in the paper.
- SYN3: four peaks are imported at random time points of the time series with random noises, which keep the orders in two dimensions, and different order means different class. Noises follow the standard random distribution, and peaks are randomly chosen from 10 and -10. Each class always keeps the same order of peaks. Example figures can be found in the paper.

These datasets can be created by the functions in `syntheticGenerator.py` (`generateSYN1`, `generateSYN2`, `generateSYN3`). The example run can be found in `example3.py`.

# Scripts for competitor codes

We directly use the authors' code with the same experiment settings. We provide our code snippets to replicate the experiment settings (parameter search). These can be found in each folder (`MR-PETSC.py` in `PETSC` and `XEM.py` in `XEM`).
To use these two scripts, the original code is required to be placed in the folders together with the scripts.
Our scripts directly run the original code, thus the original instruction to set up the algorithms should be followed prior to running our scripts.

## Repository

We refer to the original repository and please follow the instruction in the original implementation to set up the algorithms. 

- PETSC: https://bitbucket.org/len_feremans/petsc
- XEM: https://github.com/XAIseries/XEM

To run our scripts (`PETSC.py` and `XEM.py`), please clone or copy the whole repository into the designated folders `PETSC` and `XEM`. Since the original implementations has strong dependency on their folder structure, the scripts only work if they are in the folder together with original code. If needed, please follow initial setting process and adjust to the dependencies given by the original code as well.
## Input

- PETSC: The author's implementation requires sktime's ts format. This can be downloaded directly from the UCR/UEA repository. Note that PETSC has a strict dependency on a previous version of sktime.
- XEM: The author's implementation requires a specific `parquet` format. Please refer to the original repository for more information of how to create the files.

XEM requires further setting in `config.yml` file in the `config` folder. We modify the configuration file to support the grid search of the parameters. Each parameter (window, trees, max_depth, max_samples) receives a list of parameters the grid search explores, not a single value. We have put one example of parquet format (`BasicMotions`) with its configuration file. 

## How to run

You can run both code without any arguments directly with Python in the console. 
XEM's input data is controlled by `config.xml` and PETSC's input data can be changed in the code, in the list `selection`.
# Plots

All our plots used in the paper are also available in a pdf form in the `plots` folder.

# Results

All the numeric results are available in our repository. In the `result` folder, there are four files as follows:

- `multivariate_effectiveness.csv`: Effectiveness (classification accuracy) benchmark data for three interpretable multivariate classifiers on the 26 UEA multivariate time series datasets. 
- `multivariate_efficiency.csv`: Efficiency (runtime) benchmark data for three interpretable multivariate classifiers and four non-interpretable multivariate classifiers on the 26 UEA multivariate time series datasets, with training and test times separately. 
- `univariate_effectiveness.csv`: Effectiveness (classification accuracy) benchmark data for three interpretable time series classifiers on the 112 UCR multivariate time series datasets. 
- `univariate_efficiency.csv`: Efficiency (runtime) benchmark data for three interpretable time series classifiers on the 112 UCR multivariate time series datasets on a single core. 

We provide both training and test times separately for runtime on multivariate datasets and also provide runtime values for four non-interpretable classifiers.
Any blank cell means an algorithm cannot complete on a specific dataset within the runtime or memory cutoff.

For effectiveness of non-interpretable classifiers, we have brought the result from the most recent benchmark in [HIVE-COTE](https://link.springer.com/article/10.1007/s10994-021-06057-9) paper.
