"""
This script has been modified from the original PETSC code to support randomized search
to replicate the experiment settings in the paper
"""

import os
import warnings
import sys
import pickle
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sklearn.model_selection import train_test_split

# PETSC imports - make sure PETSC repository as well as commom.py is placed 
from common import mr_petsc_multivariate, copy_ts_mv_single_file

warnings.filterwarnings('ignore')
best_params_col = {}
dir = 'data'
selection = ['BasicMotions']

def grid_search_mr_petsc_multivariate(train_filename, test_filename):
    print(">>MR-PETSC-MULTIVARIATE parameter optimisation")
    # compute minsize
    train_x, train_y = load_from_tsfile_to_dataframe(train_filename)
    nr_series = train_y.shape[0]
    minlen = 10000000
    for i in range(0, nr_series):
        len_i = train_x['dim_0'].iloc[i].to_numpy().shape[0]
        minlen = min(minlen, len_i)
    # random search
    best_err = 1.0
    stride = 1
    best_params = {}
    for k in [100,250,500]:
        for alphabet in [4,8,12]:
            for rdur in [1.0,1.1,1.5]:
                for w in [10,15,20]:
                    print(k, alphabet, rdur, w)
                    if k == 250 and alphabet==4 and rdur==1.1 and w == 15:
                        print("X ", end='') 

                    err = mr_petsc_multivariate(train_filename, test_filename, w, alphabet, k, min_len=w//3, rdur=rdur, stride=stride)
                    if err < best_err:
                        best_err = err
                        best_params = {'w': w,
                                        'alphabet': alphabet,
                                        'k': k,
                                        'rdur': rdur}
                        print(best_err, best_params)

    return best_params



def run_mr_petsc_multivariate_parameter_optimisation(train_filename, test_filename):
    # load data for training
    train_x, train_y = load_from_tsfile_to_dataframe(train_filename)
    test_x, test_y = load_from_tsfile_to_dataframe(test_filename)
    
    # Parameter optimalisation
    train_valid_x, test_valid_x, train_valid_y, test_valid_y = train_test_split(train_x, train_y, test_size=0.33)

    fname_validation_train = train_filename + '_validation_train_.ts'
    fname_validation_test = train_filename + '_validation_test_.ts'
    copy_ts_mv_single_file(train_filename, fname_validation_train, train_valid_x, train_valid_y)
    copy_ts_mv_single_file(train_filename, fname_validation_test, test_valid_x, test_valid_y)
    # search:
    best_params = grid_search_mr_petsc_multivariate(fname_validation_train, fname_validation_test)
    
    # Run PETSC outer loop
    print(best_params)

    err = mr_petsc_multivariate(train_filename, test_filename, best_params['w'], best_params['alphabet'],
               best_params['k'], int(best_params['w']/3), best_params['rdur'], stride=1)
    print('MR-PETSC-MULTIVARIATE error on {:} is {:.3f} with parameters {:}'.format(os.path.basename(train_filename), err, best_params))
    best_params_col[train_filename] = {"best_params": best_params, "error": err}
    with open("result"+sys.argv[1]+".pickle", "wb") as output_file:
        pickle.dump(best_params_col, output_file)
   
    return err

dct_results_folded = {}
for dataset in selection: #selection_saxvsm:
    dct_results_folded[dataset] = {}
    
print("MR-PETSC-MULTIVARIATE ramdom search params") 

folds = 10

for i in range(0,folds): 
    dct_results = {}
    for dataset in selection:
        print(dataset)

        train_filename = os.path.join(dir, dataset) + '/' + dataset + '_TRAIN.ts' 
        test_filename = os.path.join(dir, dataset) + '/' + dataset + '_TEST.ts' 
        # Single run
        #err =  mr_petsc_multivariate(train_filename, test_filename, w=2, alphabet=4, k=200, min_len=1, rdur=1.1, stride=1)
        # Grid search
        err =  run_mr_petsc_multivariate_parameter_optimisation(train_filename, test_filename)
        
        dct_results[dataset] = err
        dct_results_folded[dataset][i] = err
        
# print(dct_results_folded)
    
