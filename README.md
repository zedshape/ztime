# Z-Time: Efficient and Effective Interpretable Multivariate Time Series Classification

Z-Time is an efficient and effective interpretable multivariate time series classifier. It employs temporal abstractions and builds temporal relations of event intervals to create interpretable features across multiple time series dimensions.

# Dependencies
This application needs the latest version of the following packages (in December 2022) and Python >= 3.8. We restrict the usage of the libraries for clear reproducibility of the research. For details, please refer to `requirements.txt`.

`numpy`, `pandas`, `pyts`, `scipy`

Numpy, pandas, and scipy are used for dataset handling. Pyts is used for temporal abstractions.

# How to use

Our helper.py focuses on automating the process of reproducing the experiment. The experiments can be reproducible by mainly using the following functions:

`simpleTrial`: Run one trial of Z-Time classification given parameters.
  - data: variable that includes both training and test sets.
  - alphabet_size: $\lambda_1$ parameter.
  - alphabet_size_slope: $\lambda_2$ parameter.
  - window_size: $w$ parameter.
  - split_no: $p$ parameter.
  - step: $\sigma$ parameter.
  - multivariate: indication if the dataset is multivariate. The default is `True`.

`randomizedSearch`: Run a randomized search of $k$ iterations.
  - data: variable that includes both training and test sets.
  - steps: list of $\sigma$ parameters.
  - alphabet_sizes: list of $\lambda_1$ parameters.
  - alphabet_sizes_slope: list of $\lambda_2$ parameters.
  - window_sizes: list of $w$ parameters.
  - split_nos: list of $p$ parameters.
  - cycle: the number of iterations for randomized search.
  - multivariate: indication if the dataset is multivariate. The default is `True`.

Please refer to the paper for a detailed explanation of the parameters.

Note that the dataset should follow the format of a 3D numpy array with the shape (size, dimension, length). We have one instance incorporating both training and test sets. The data structure should be as follows to run Z-Time using the helper.py:

- data["TRAIN"]["X"]: Training set in 3D numpy array format with the shape (size, dimension, length)
- data["TRAIN"]["y"]: List of training labels
- data["TEST"]["X"]: Training set in 3D numpy array format with the shape (size, dimension, length)
- data["TEST"]["y"]: List of test labels

The example of `simpleTrial` is available in `example.py`. The example of `randomizedSearch` is available in `example2.py`.

# Datasets

## The UCR/UEA time series datasets

We use the multivariate and univariate time series datasets in the UCR/UEA repository. These datasets can be found at [here](http://www.timeseriesclassification.com). Z-Time uses a 3D numpy array form of those datasets with the shape (size, dimension, length). For your reference, we have a converted BasicMotions dataset in our repository. 

## Synthetic datasets

In our experiment, we use three synthetic datasets (SYN1, SYN2, and SYN3), and they are all available in a pickled form in the `data` folder. You can use the `simpleTrial` function in the same way as other datasets. Here are details of our synthetic data.

- SYN1: two sinusoidal concave/convex patterns are imported in the first and last 25% of the time series with random noises. Sinusoidal patterns have a peak size of 2 or -2. Example figures can be found in the paper.
- SYN2: two peaks are imported at random time points of the time series with random noises, which keep the orders in two dimensions, and different order means different class. The dataset has two classes. Noises follow the standard random distribution, and peaks are randomly chosen from 10 and -10. Each class always keeps the same order of peaks. Example figures can be found in the paper.
- SYN3: four peaks are imported at random time points of the time series with random noises, which keep the orders in two dimensions, and different order means different class. Noises follow the standard random distribution, and peaks are randomly chosen from 10 and -10. Each class always keeps the same order of peaks. Example figures can be found in the paper.

These datasets can be created by the functions in `syntheticGenerator.py` (`generateSYN1`, `generateSYN2`, `generateSYN3`). The example run can be found in `example3.py`.

# Plots

All our plots used in the paper are also available in a pdf form in the `plots` folder.

# Results

All the numeric results are available in our repository. In the `result` folder, there are four files as follows:

- `multivariate_effectiveness.csv`: Effectiveness (classification accuracy) benchmark data for three interpretable multivariate classifiers on the 26 UEA multivariate time series datasets. 
- `multivariate_efficiency.csv`: Efficiency (runtime) benchmark data for three interpretable multivariate classifiers on the 26 UEA multivariate time series datasets. 
- `univariate_effectiveness.csv`: Effectiveness (classification accuracy) benchmark data for three interpretable time series classifiers on the 112 UCR multivariate time series datasets. 
- `univariate_efficiency.csv`: Efficiency (runtime) benchmark data for three interpretable time series classifiers on the 112 UCR multivariate time series datasets on a single core. 

Any blank cell means an algorithm cannot complete on a specific dataset within the runtime or memory cutoff.

For non-interpretable classifiers, we have brought the result from the most recent benchmark in the [HIVE-COTE 2.0](https://link.springer.com/article/10.1007/s10994-021-06057-9) paper.
