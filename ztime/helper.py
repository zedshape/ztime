"""
Z-Time helper module
=====================
This helper module has functions that help create interval structure from raw time series.
"""

from pyts.approximation import SymbolicAggregateApproximation, PiecewiseAggregateApproximation
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifierCV
import time
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from . import ZTime

def znorm(samples):
  return (samples - samples.mean(axis=-1)[:,...,np.newaxis]) / samples.std(axis=-1)[:,...,np.newaxis]

def createDatabase(data, window_size = 10, window_size_slope = 5, alphabet_size = 3, alphabet_size_slope = 5, glob=False, uniform=True, quantile=True, normal=True):
    alphabet_origin = 97
    
    alphabets_1 = [chr(alphabet_origin + i) for i in range(alphabet_size)]
    alphabets_2 = [chr(alphabet_origin + i + alphabet_size) for i in range(alphabet_size)]
    alphabets_3 = [chr(alphabet_origin + i + alphabet_size*2) for i in range(alphabet_size)]
    alphabets_4 = [chr(alphabet_origin + i + alphabet_size*3) for i in range(alphabet_size)]
    alphabets_5 = [chr(alphabet_origin + i + alphabet_size*4) for i in range(alphabet_size)]
    alphabets_6 = [chr(alphabet_origin + i + alphabet_size*5) for i in range(alphabet_size)]
    alphabets_7 = [chr(alphabet_origin + i + alphabet_size*6) for i in range(alphabet_size_slope)]
    alphabets_8 = [chr(alphabet_origin + i + alphabet_size*6 + alphabet_size_slope*1) for i in range(alphabet_size_slope)]
    alphabets_9 = [chr(alphabet_origin + i + alphabet_size*6 + alphabet_size_slope*2) for i in range(alphabet_size_slope)]
    alphabets_10 = [chr(alphabet_origin + i + alphabet_size*6 + alphabet_size_slope*3) for i in range(alphabet_size_slope)]

    PAA_transformer = PiecewiseAggregateApproximation(window_size=window_size)

    SAX_uniform = SAXify(data,  alphabet = alphabets_1, n_bins = alphabet_size, glob=glob, strategy='uniform')
    SAX_quantile = SAXify(data,  alphabet = alphabets_2, n_bins = alphabet_size, glob=glob, strategy='quantile')
    SAX_normal = SAXify(data,  alphabet = alphabets_3, n_bins = alphabet_size, glob=glob, strategy='normal')
    
    PAA_SAX_uniform = SAXify(PAA_transformer.transform(data), alphabet = alphabets_4, n_bins = alphabet_size, glob=glob, strategy='uniform')
    PAA_SAX_quantile = SAXify(PAA_transformer.transform(data), alphabet = alphabets_5, n_bins = alphabet_size, glob=glob, strategy='quantile')
    PAA_SAX_normal = SAXify(PAA_transformer.transform(data), alphabet = alphabets_6, n_bins = alphabet_size, glob=glob, strategy='normal')

    diff_data = np.diff(data, prepend=0)
    diff_double_data = np.diff(diff_data, prepend=0)
    diff_SAX_uniform = SAXify(diff_data,  alphabet = alphabets_7, n_bins = alphabet_size_slope, glob=glob, strategy='uniform')
    diff_SAX_quantile = SAXify(diff_data,  alphabet = alphabets_8, n_bins = alphabet_size_slope, glob=glob, strategy='quantile')
    diff_double_uniform_SAX = SAXify(diff_double_data, alphabet = alphabets_9, n_bins = alphabet_size_slope, glob=glob, strategy='uniform')
    diff_double_quantile_SAX = SAXify(diff_double_data, alphabet = alphabets_10, n_bins = alphabet_size_slope, glob=glob, strategy='quantile')

    new_database = []
    
    for row in range(data.shape[0]):
        newRow = []

        newRow += createIntervals(diff_SAX_quantile[row], end_size = diff_data.shape[-1], window_size = window_size_slope)
        newRow += createIntervals(diff_double_quantile_SAX[row], end_size = diff_double_data.shape[-1], window_size = window_size_slope)
        newRow += createIntervals(diff_SAX_uniform[row], end_size = diff_data.shape[-1], window_size = window_size_slope)
        newRow += createIntervals(diff_double_uniform_SAX[row], end_size = diff_double_data.shape[-1], window_size = window_size_slope)

        
        if normal == True:
            newRow += createIntervals(SAX_normal[row],  end_size = data.shape[-1], window_size = 1)
            newRow += createIntervals(PAA_SAX_normal[row], end_size = data.shape[-1], window_size = window_size)
        if uniform == True:
            newRow += createIntervals(SAX_uniform[row],  end_size = data.shape[-1], window_size = 1)
            newRow += createIntervals(PAA_SAX_uniform[row], end_size = data.shape[-1], window_size = window_size)
        if quantile == True:
            newRow += createIntervals(SAX_quantile[row],  end_size = data.shape[-1], window_size = 1)
            newRow += createIntervals(PAA_SAX_quantile[row], end_size = data.shape[-1], window_size = window_size)

        new_database.append(removeSpecificEventLabel(newRow, []))
    return new_database

def createIntervalsMissingInterval(data, mask, end_size, window_size=1):
    prevCol = None
    count = 0
    prevFinalCount = 0
    newRow = []
    
    if window_size != 1:
        data = np.repeat(data, window_size)

    for colIdx in range(len(data)):
        currentCol = data[colIdx]

        if prevCol == None:
            prevCol = currentCol
            count += 1
        else:
            if currentCol == prevCol:
                count += 1
            else:
                if prevColMask == False:
                    newRow.append(((prevFinalCount), prevFinalCount + (count), prevCol))
                prevFinalCount = (prevFinalCount + count)+1
                count = 1
                prevCol = currentCol
        prevColMask = mask[colIdx]

    if prevColMask == False:
        newRow.append((prevFinalCount, end_size-1, prevCol))
    
    return newRow

def createIntervals(data, end_size, window_size=1):
    prevCol = None
    count = 0
    prevFinalCount = 0
    newRow = []
    if window_size != 1:
        data = np.repeat(data, window_size)
    for currentCol in data:
        # Initial condition
        if prevCol == None:
            prevCol = currentCol
            # count += 1
        else:
            if currentCol == prevCol:
                count += 1
            else:
                newRow.append(((prevFinalCount), prevFinalCount + (count), prevCol))
                prevFinalCount = (prevFinalCount + count)+1
                count = 0
                prevCol = currentCol

    newRow.append((prevFinalCount, end_size-1, prevCol))
    
    return newRow

def createMultivariateDatabaseVL(data, window_size = 10, window_size_slope = 5, alphabet_size = 3, alphabet_size_slope = 5, glob=False, uniform=True, quantile=True, normal=True, varyingLength=False):

    dims = len(data[0])
    diff_data = np.zeros(1)
    diff_double_data = np.zeros(1)

    SAX_PAA_info = {}
    step_size = alphabet_size*6 + alphabet_size_slope*4
    for dim in range(dims):

        data_transformed = [i[dim] for i in data]

        alphabets_1 = [step_size*dim + i for i in range(alphabet_size)] #0,1,2
        alphabets_2 = [step_size*dim + i + alphabet_size for i in range(alphabet_size)] #3,4,5
        alphabets_3 = [step_size*dim + i + alphabet_size*2 for i in range(alphabet_size)] #6,7,8
        alphabets_4 = [step_size*dim + i + alphabet_size*3 for i in range(alphabet_size)] #9,10,11
        alphabets_5 = [step_size*dim + i + alphabet_size*4 for i in range(alphabet_size)] #12,13,14
        alphabets_6 = [step_size*dim + i + alphabet_size*5 for i in range(alphabet_size)] #15,16,17

        # FOR SLOPES
        alphabets_7 = [step_size*dim  + i + alphabet_size*6 for i in range(alphabet_size_slope)] #18, 19
        alphabets_8 = [step_size*dim  + i + alphabet_size*6 + alphabet_size_slope*1 for i in range(alphabet_size_slope)] #20,21
        alphabets_9 = [step_size*dim  + i + alphabet_size*6 + alphabet_size_slope*2 for i in range(alphabet_size_slope)] #22,23
        alphabets_10 = [step_size*dim  + i + alphabet_size*6 + alphabet_size_slope*3 for i in range(alphabet_size_slope)] #24,25

        PAA_transformer = PiecewiseAggregateApproximation(window_size=window_size)

        diff_data = [np.diff(i, prepend=i[0]) for i in data_transformed]
        diff_double_data = [np.diff(i, prepend=i[0]) for i in diff_data]

        SAX_uniform = [SAXify([a], alphabet = alphabets_1, n_bins = alphabet_size, glob=glob, strategy='uniform')[0] for a in data_transformed]
        SAX_quantile = [SAXify([a], alphabet = alphabets_2, n_bins = alphabet_size, glob=glob, strategy='quantile')[0] for a in data_transformed]
        SAX_normal = [SAXify([a], alphabet = alphabets_3, n_bins = alphabet_size, glob=glob, strategy='normal')[0] for a in data_transformed]
        
        PAA_SAX_uniform = [SAXify(PAA_transformer.transform([a]), alphabet = alphabets_4, n_bins = alphabet_size, glob=glob, strategy='uniform')[0] for a in data_transformed]
        PAA_SAX_quantile = [SAXify(PAA_transformer.transform([a]), alphabet = alphabets_5, n_bins = alphabet_size, glob=glob, strategy='quantile')[0] for a in data_transformed]
        PAA_SAX_normal = [SAXify(PAA_transformer.transform([a]), alphabet = alphabets_6, n_bins = alphabet_size, glob=glob, strategy='normal')[0] for a in data_transformed]

        diff_SAX_uniform = [SAXify([a], alphabet = alphabets_7, n_bins = alphabet_size_slope, glob=glob, strategy='uniform')[0] for a in diff_data]
        diff_SAX_quantile = [SAXify([a], alphabet = alphabets_8, n_bins = alphabet_size_slope, glob=glob, strategy='quantile')[0] for a in diff_data]

        diff_double_uniform_SAX = [SAXify([a], alphabet = alphabets_9, n_bins = alphabet_size_slope, glob=glob, strategy='uniform')[0] for a in diff_double_data]
        diff_double_quantile_SAX = [SAXify([a], alphabet = alphabets_10, n_bins = alphabet_size_slope, glob=glob, strategy='quantile')[0] for a in diff_double_data]

        SAX_PAA_info[dim] = {"SAX_uniform": SAX_uniform, 
                             "SAX_quantile": SAX_quantile, 
                             "SAX_normal": SAX_normal, 
                             "PAA_SAX_uniform": PAA_SAX_uniform, 
                             "PAA_SAX_quantile": PAA_SAX_quantile, 
                             "PAA_SAX_normal": PAA_SAX_normal, 
                             "diff_SAX_uniform": diff_SAX_uniform, 
                             "diff_SAX_quantile": diff_SAX_quantile, 
                             "diff_double_uniform_SAX": diff_double_uniform_SAX, 
                             "diff_double_quantile_SAX": diff_double_quantile_SAX}

    new_database = []
    
    for row in range(len(data)):

        newRow = []

        for dim in range(dims):

            newRow += createIntervals(SAX_PAA_info[dim]["diff_SAX_quantile"][row], end_size = len(data[row][0]), window_size = window_size_slope)
            newRow += createIntervals(SAX_PAA_info[dim]["diff_double_quantile_SAX"][row], end_size = len(data[row][0]), window_size = window_size_slope)
            newRow += createIntervals(SAX_PAA_info[dim]["diff_SAX_uniform"][row], end_size =  len(data[row][0]), window_size = window_size_slope)
            newRow += createIntervals(SAX_PAA_info[dim]["diff_double_uniform_SAX"][row], end_size = len(data[row][0]), window_size = window_size_slope)

            if normal == True:
                newRow += createIntervals(SAX_PAA_info[dim]["SAX_normal"][row],  end_size = len(data[row][0]), window_size = 1)
                newRow += createIntervals(SAX_PAA_info[dim]["PAA_SAX_normal"][row], end_size = len(data[row][0]), window_size = window_size)
            if uniform == True:
                newRow += createIntervals(SAX_PAA_info[dim]["SAX_uniform"][row],  end_size =  len(data[row][0]), window_size = 1)
                newRow += createIntervals(SAX_PAA_info[dim]["PAA_SAX_uniform"][row], end_size = len(data[row][0]), window_size = window_size)
            if quantile == True:
                newRow += createIntervals(SAX_PAA_info[dim]["SAX_quantile"][row],  end_size = len(data[row][0]), window_size = 1)
                newRow += createIntervals(SAX_PAA_info[dim]["PAA_SAX_quantile"][row], end_size = len(data[row][0]), window_size = window_size)
            
        new_database.append(sorted(newRow))
    return new_database


def createMultivariateDatabaseMissingData(data, mask, window_size = 10, window_size_slope = 5, alphabet_size = 3, alphabet_size_slope = 5, glob=False, uniform=True, quantile=True, normal=True, varyingLength=False):
    alphabet_origin = 97
    
    # get dimension
    # row, dimension, length
    dims = data.shape[1]
    diff_data = np.zeros(1)
    diff_double_data = np.zeros(1)
    data_transformed = np.zeros(1)

    SAX_PAA_info = {}

    
    step_size = alphabet_size*6 + alphabet_size_slope*4
    # for each dimension we need to save SAX_PAA information
    for dim in range(dims):
        #transform the data into the understandable form
        
        #data_transformed = np.array([i.to_numpy() for i in data[dim]])
        data_transformed = data[:, dim, :]

        alphabets_1 = [step_size*dim + i for i in range(alphabet_size)] #0,1,2
        alphabets_2 = [step_size*dim + i + alphabet_size for i in range(alphabet_size)] #3,4,5
        alphabets_3 = [step_size*dim + i + alphabet_size*2 for i in range(alphabet_size)] #6,7,8
        alphabets_4 = [step_size*dim + i + alphabet_size*3 for i in range(alphabet_size)] #9,10,11
        alphabets_5 = [step_size*dim + i + alphabet_size*4 for i in range(alphabet_size)] #12,13,14
        alphabets_6 = [step_size*dim + i + alphabet_size*5 for i in range(alphabet_size)] #15,16,17

        # FOR SLOPES
        alphabets_7 = [step_size*dim  + i + alphabet_size*6 for i in range(alphabet_size_slope)] #18, 19
        alphabets_8 = [step_size*dim  + i + alphabet_size*6 + alphabet_size_slope*1 for i in range(alphabet_size_slope)] #20,21
        alphabets_9 = [step_size*dim  + i + alphabet_size*6 + alphabet_size_slope*2 for i in range(alphabet_size_slope)] #22,23
        alphabets_10 = [step_size*dim  + i + alphabet_size*6 + alphabet_size_slope*3 for i in range(alphabet_size_slope)] #24,25

        PAA_transformer = PiecewiseAggregateApproximation(window_size=window_size)

        # diff the dataset
        diff_data = np.diff(data_transformed, prepend=0)
        diff_double_data = np.diff(diff_data, prepend=0)
        
        # Step 1. Create PAA-SAX intervals
        #
        # Raw dataset will be PAA-ed to catch the pattern robust to outliers and noises
        #

        
        SAX_uniform = SAXify(data_transformed,  alphabet = alphabets_1, n_bins = alphabet_size, glob=glob, strategy='uniform')
        SAX_quantile = SAXify(data_transformed,  alphabet = alphabets_2, n_bins = alphabet_size, glob=glob, strategy='quantile')
        SAX_normal = SAXify(data_transformed,  alphabet = alphabets_3, n_bins = alphabet_size, glob=glob, strategy='normal')
        
        PAA_SAX_uniform = SAXify(PAA_transformer.transform(data_transformed), alphabet = alphabets_4, n_bins = alphabet_size, glob=glob, strategy='uniform')
        PAA_SAX_quantile = SAXify(PAA_transformer.transform(data_transformed), alphabet = alphabets_5, n_bins = alphabet_size, glob=glob, strategy='quantile')
        PAA_SAX_normal = SAXify(PAA_transformer.transform(data_transformed), alphabet = alphabets_6, n_bins = alphabet_size, glob=glob, strategy='normal')

        # Step 2. Create only-SAX intervals
        #
        # derivatives will not be PAA-ed since they only have limited numbers
        # For the same reason diff does not have normal layer since it is too strong assumption
        
        # diff does not have normal layer since it is too strong assumption
        diff_SAX_uniform = SAXify(diff_data,  alphabet = alphabets_7, n_bins = alphabet_size_slope, glob=glob, strategy='uniform')
        diff_SAX_quantile = SAXify(diff_data,  alphabet = alphabets_8, n_bins = alphabet_size_slope, glob=glob, strategy='quantile')
        
        # double diff does not have normal layer since it is too strong assumption
        diff_double_uniform_SAX = SAXify(diff_double_data, alphabet = alphabets_9, n_bins = alphabet_size_slope, glob=glob, strategy='uniform')
        diff_double_quantile_SAX = SAXify(diff_double_data, alphabet = alphabets_10, n_bins = alphabet_size_slope, glob=glob, strategy='quantile')

        SAX_PAA_info[dim] = {"SAX_uniform": SAX_uniform, 
                             "SAX_quantile": SAX_quantile, 
                             "SAX_normal": SAX_normal, 
                             "PAA_SAX_uniform": PAA_SAX_uniform, 
                             "PAA_SAX_quantile": PAA_SAX_quantile, 
                             "PAA_SAX_normal": PAA_SAX_normal, 
                             "diff_SAX_uniform": diff_SAX_uniform, 
                             "diff_SAX_quantile": diff_SAX_quantile, 
                             "diff_double_uniform_SAX": diff_double_uniform_SAX, 
                             "diff_double_quantile_SAX": diff_double_quantile_SAX}

    new_database = []
    
    for row in range(data.shape[0]):
        # Each row, each dimension 
        newRow = []

        for dim in range(dims):
            # We need to add those different intervals in the same database
            
            newRow += createIntervalsMissingInterval(SAX_PAA_info[dim]["diff_SAX_quantile"][row], mask[row][dim], end_size = diff_data.shape[-1], window_size = window_size_slope)
            newRow += createIntervalsMissingInterval(SAX_PAA_info[dim]["diff_double_quantile_SAX"][row],mask[row][dim],  end_size = diff_double_data.shape[-1], window_size = window_size_slope)
            newRow += createIntervalsMissingInterval(SAX_PAA_info[dim]["diff_SAX_uniform"][row],mask[row][dim],  end_size = diff_data.shape[-1], window_size = window_size_slope)
            newRow += createIntervalsMissingInterval(SAX_PAA_info[dim]["diff_double_uniform_SAX"][row],mask[row][dim],  end_size = diff_double_data.shape[-1], window_size = window_size_slope)

            
            if normal == True:
                newRow += createIntervalsMissingInterval(SAX_PAA_info[dim]["SAX_normal"][row],mask[row][dim],   end_size = data_transformed.shape[-1], window_size = 1)
                newRow += createIntervals(SAX_PAA_info[dim]["PAA_SAX_normal"][row], end_size = data_transformed.shape[-1], window_size = window_size)
            if uniform == True:
                newRow += createIntervalsMissingInterval(SAX_PAA_info[dim]["SAX_uniform"][row], mask[row][dim],  end_size = data_transformed.shape[-1], window_size = 1)
                newRow += createIntervals(SAX_PAA_info[dim]["PAA_SAX_uniform"][row], end_size = data_transformed.shape[-1], window_size = window_size)
            if quantile == True:
                newRow += createIntervalsMissingInterval(SAX_PAA_info[dim]["SAX_quantile"][row],mask[row][dim],   end_size = data_transformed.shape[-1], window_size = 1)
                newRow += createIntervals(SAX_PAA_info[dim]["PAA_SAX_quantile"][row], end_size = data_transformed.shape[-1], window_size = window_size)
            
        new_database.append(sorted(newRow))
    return new_database

def createMultivariateDatabase(data, window_size = 10, window_size_slope = 5, alphabet_size = 3, alphabet_size_slope = 5, glob=False, uniform=True, quantile=True, normal=True, varyingLength=False):
    alphabet_origin = 97
    
    dims = data.shape[1]
    diff_data = np.zeros(1)
    diff_double_data = np.zeros(1)
    data_transformed = np.zeros(1)

    SAX_PAA_info = {}

    
    step_size = alphabet_size*6 + alphabet_size_slope*4
    for dim in range(dims):

        data_transformed = data[:, dim, :]

        alphabets_1 = [step_size*dim + i for i in range(alphabet_size)] #0,1,2
        alphabets_2 = [step_size*dim + i + alphabet_size for i in range(alphabet_size)] #3,4,5
        alphabets_3 = [step_size*dim + i + alphabet_size*2 for i in range(alphabet_size)] #6,7,8
        alphabets_4 = [step_size*dim + i + alphabet_size*3 for i in range(alphabet_size)] #9,10,11
        alphabets_5 = [step_size*dim + i + alphabet_size*4 for i in range(alphabet_size)] #12,13,14
        alphabets_6 = [step_size*dim + i + alphabet_size*5 for i in range(alphabet_size)] #15,16,17
        alphabets_7 = [step_size*dim  + i + alphabet_size*6 for i in range(alphabet_size_slope)] #18, 19
        alphabets_8 = [step_size*dim  + i + alphabet_size*6 + alphabet_size_slope*1 for i in range(alphabet_size_slope)] #20,21
        alphabets_9 = [step_size*dim  + i + alphabet_size*6 + alphabet_size_slope*2 for i in range(alphabet_size_slope)] #22,23
        alphabets_10 = [step_size*dim  + i + alphabet_size*6 + alphabet_size_slope*3 for i in range(alphabet_size_slope)] #24,25

        PAA_transformer = PiecewiseAggregateApproximation(window_size=window_size)

        diff_data = np.diff(data_transformed, prepend=0)
        diff_double_data = np.diff(diff_data, prepend=0)

        
        SAX_uniform = SAXify(data_transformed,  alphabet = alphabets_1, n_bins = alphabet_size, glob=glob, strategy='uniform')
        SAX_quantile = SAXify(data_transformed,  alphabet = alphabets_2, n_bins = alphabet_size, glob=glob, strategy='quantile')
        SAX_normal = SAXify(data_transformed,  alphabet = alphabets_3, n_bins = alphabet_size, glob=glob, strategy='normal')
        
        PAA_SAX_uniform = SAXify(PAA_transformer.transform(data_transformed), alphabet = alphabets_4, n_bins = alphabet_size, glob=glob, strategy='uniform')
        PAA_SAX_quantile = SAXify(PAA_transformer.transform(data_transformed), alphabet = alphabets_5, n_bins = alphabet_size, glob=glob, strategy='quantile')
        PAA_SAX_normal = SAXify(PAA_transformer.transform(data_transformed), alphabet = alphabets_6, n_bins = alphabet_size, glob=glob, strategy='normal')
  
        diff_SAX_uniform = SAXify(diff_data,  alphabet = alphabets_7, n_bins = alphabet_size_slope, glob=glob, strategy='uniform')
        diff_SAX_quantile = SAXify(diff_data,  alphabet = alphabets_8, n_bins = alphabet_size_slope, glob=glob, strategy='quantile')
        
        diff_double_uniform_SAX = SAXify(diff_double_data, alphabet = alphabets_9, n_bins = alphabet_size_slope, glob=glob, strategy='uniform')
        diff_double_quantile_SAX = SAXify(diff_double_data, alphabet = alphabets_10, n_bins = alphabet_size_slope, glob=glob, strategy='quantile')

        SAX_PAA_info[dim] = {"SAX_uniform": SAX_uniform, 
                             "SAX_quantile": SAX_quantile, 
                             "SAX_normal": SAX_normal, 
                             "PAA_SAX_uniform": PAA_SAX_uniform, 
                             "PAA_SAX_quantile": PAA_SAX_quantile, 
                             "PAA_SAX_normal": PAA_SAX_normal, 
                             "diff_SAX_uniform": diff_SAX_uniform, 
                             "diff_SAX_quantile": diff_SAX_quantile, 
                             "diff_double_uniform_SAX": diff_double_uniform_SAX, 
                             "diff_double_quantile_SAX": diff_double_quantile_SAX}

    new_database = []
    
    for row in range(data.shape[0]):

        newRow = []

        for dim in range(dims):
            
            newRow += createIntervals(SAX_PAA_info[dim]["diff_SAX_quantile"][row], end_size = diff_data.shape[-1], window_size = window_size_slope)
            newRow += createIntervals(SAX_PAA_info[dim]["diff_double_quantile_SAX"][row], end_size = diff_double_data.shape[-1], window_size = window_size_slope)
            newRow += createIntervals(SAX_PAA_info[dim]["diff_SAX_uniform"][row], end_size = diff_data.shape[-1], window_size = window_size_slope)
            newRow += createIntervals(SAX_PAA_info[dim]["diff_double_uniform_SAX"][row], end_size = diff_double_data.shape[-1], window_size = window_size_slope)

            if normal == True:
                newRow += createIntervals(SAX_PAA_info[dim]["SAX_normal"][row],  end_size = data_transformed.shape[-1], window_size = 1)
                newRow += createIntervals(SAX_PAA_info[dim]["PAA_SAX_normal"][row], end_size = data_transformed.shape[-1], window_size = window_size)
            if uniform == True:
                newRow += createIntervals(SAX_PAA_info[dim]["SAX_uniform"][row],  end_size = data_transformed.shape[-1], window_size = 1)
                newRow += createIntervals(SAX_PAA_info[dim]["PAA_SAX_uniform"][row], end_size = data_transformed.shape[-1], window_size = window_size)
            if quantile == True:
                newRow += createIntervals(SAX_PAA_info[dim]["SAX_quantile"][row],  end_size = data_transformed.shape[-1], window_size = 1)
                newRow += createIntervals(SAX_PAA_info[dim]["PAA_SAX_quantile"][row], end_size = data_transformed.shape[-1], window_size = window_size)
            
        new_database.append(sorted(newRow))
    return new_database
      
def SAXify(data, alphabet,  n_bins = 5, glob=False, strategy='uniform'):
    sax = SymbolicAggregateApproximation(n_bins = n_bins, alphabet = alphabet, strategy=strategy)
    
    if glob == True:
        globalvals = np.concatenate(data)
        data_new = sax.fit_transform([globalvals])
        data_new = data_new.reshape(data.shape)
    else:
        data_new = sax.fit_transform(data)
    return data_new

def removeSpecificEventLabel(data, labels):
    newData = []
    for row in data:
        if row[0] not in labels:
            newData.append(row)
    return newData

def checkLabelBalance(labels):
    return np.unique(labels, return_counts = True)[1] / len(labels)

def trialSequence_multivariate(X_train, X_test, y_train, y_test, window_size = 10, window_size_slope = 5, 
                    alphabet_size = 5, alphabet_size_slope = 1, 
                    glob=False, step=0, split_no = 1, varyingLength = False, missingDataTest = False,
                    mask_train = False, mask_test = False, classifier='Ridge', debug=False):

    t1 = time.perf_counter()
    
    vectorizer = DictVectorizer(dtype=np.uint16, sparse=True)
    
    # Training phase
    
    trainset = []
    algos = []
    if varyingLength == True:
        X_train_split = [np.array_split(i, split_no, axis=-1) for i in X_train]
        X_train_split = [[row[split] for row in X_train_split] for split in range(split_no)]

        if len(X_train_split[0][0][0]) < window_size:
            # NO ERROR MSG IN DEBUG MODE
            # print("WINDOW SIZE IS SMALLER THAN SPLIT")
            return None, None
    else:
        X_train_split = np.array_split(X_train, split_no, axis=-1)
        if X_train_split[-1].shape[-1] < window_size:
            # NO ERROR MSG IN DEBUG MODE
            if debug == True:
                print(X_train_split[-1].shape)
                print("WINDOW SIZE IS SMALLER THAN SPLIT")
                return None, None
        
    for split_part in X_train_split:
        if varyingLength == True:
            train_dataset = createMultivariateDatabaseVL(split_part, window_size = window_size, window_size_slope = window_size_slope, alphabet_size = alphabet_size, alphabet_size_slope= alphabet_size_slope, glob=glob)
            
        else:
            if missingDataTest == False:
                train_dataset = createMultivariateDatabase(split_part, window_size = window_size, window_size_slope = window_size_slope, alphabet_size = alphabet_size, alphabet_size_slope= alphabet_size_slope, glob=glob)
            else:
                train_dataset = createMultivariateDatabaseMissingData(split_part, mask_train, window_size = window_size, window_size_slope = window_size_slope, alphabet_size = alphabet_size, alphabet_size_slope= alphabet_size_slope, glob=glob)
                
        algorithm = ZTime.ZTime(step=step)
        tsMatrix = algorithm.train(train_dataset, y_train)
        algos.append(algorithm)
        trainset_tmp = vectorizer.fit_transform([tsMatrix[i] for i in list(tsMatrix)]).T
        trainset.append(trainset_tmp)


    algorithm = None
    trainset = hstack(trainset)
    scaler = StandardScaler(with_mean = False)
    
    if classifier == 'Ridge':
        cls = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
    elif classifier == 'Elastic':
        cls = SGDClassifier(max_iter=1000, tol=1e-3, penalty='elasticnet')
    elif classifier == 'Lasso':
        cls = SGDClassifier(max_iter=1000, tol=1e-3, penalty='l1')
    else:
        cls = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
    
    trainset = scaler.fit_transform(trainset)
    cls.fit(trainset, y_train)


    t2 = time.perf_counter()
    # Testing phase
    trainset = None
    testset = []      
    # Training phase
    if varyingLength == True:
        X_test_split = [np.array_split(i, split_no, axis=-1) for i in X_test]
        X_test_split = [[row[split] for row in X_test_split] for split in range(split_no)]
    else:
        X_test_split = np.array_split(X_test, split_no, axis=-1)
    
    for (split_part, algorithm) in zip(X_test_split, algos):
        if varyingLength == True:
            test_dataset = createMultivariateDatabaseVL(split_part, window_size = window_size, window_size_slope = window_size_slope, alphabet_size = alphabet_size, alphabet_size_slope = alphabet_size_slope, glob=glob)
        else:
            if missingDataTest == False:
                test_dataset = createMultivariateDatabase(split_part, window_size = window_size, window_size_slope = window_size_slope, alphabet_size = alphabet_size, alphabet_size_slope = alphabet_size_slope, glob=glob)
            else:
                test_dataset = createMultivariateDatabaseMissingData(split_part, mask_test, window_size = window_size, window_size_slope = window_size_slope, alphabet_size = alphabet_size, alphabet_size_slope = alphabet_size_slope, glob=glob)
        
        tsMatrix_test = algorithm.test(test_dataset)
        testset_tmp = vectorizer.fit_transform([tsMatrix_test[i] for i in list(tsMatrix_test)]).T
        testset.append(testset_tmp)

    testset = hstack(testset)
    testset = scaler.transform(testset)
    lr_score = cls.score(testset, y_test)
    t3 = time.perf_counter()
    timegap = t3-t1

    if debug == True:
        print("Training time:", t2-t1, (t2-t1)/timegap)
        print("Test time:", t3-t2, (t3-t2)/timegap)
        print("Total time:",timegap)

    return lr_score, timegap

def trialSequence(X_train, X_test, y_train, y_test, 
                    window_size = 10, window_size_slope = 5, 
                    alphabet_size = 5, alphabet_size_slope = 5, 
                    glob=False, step=10, split_no=1, classifier='Ridge'):

    t1 = time.perf_counter()

    X_train_split = np.array_split(X_train, split_no, axis=1)
    X_test_split = np.array_split(X_test, split_no, axis=1)

    trainset_created = []
    testset_created = []

    if X_train_split[-1].shape[1] < window_size:
        print("WINDOW SIZE IS SMALLER THAN SPLIT")  
        return None, None
    
    for split_part in zip(X_train_split, X_test_split):
        
        ts1 = time.perf_counter()
        train_dataset = createDatabase(split_part[0], window_size = window_size, window_size_slope = window_size_slope, alphabet_size = alphabet_size, alphabet_size_slope= alphabet_size_slope, glob=glob)
        test_dataset = createDatabase(split_part[1], window_size = window_size, window_size_slope = window_size_slope, alphabet_size = alphabet_size, alphabet_size_slope= alphabet_size_slope, glob=glob)
        
        algorithm = ZTime.ZTime(step=step)
        tsMatrix = algorithm.train(train_dataset, y_train)
        
        vectorizer = DictVectorizer(dtype=np.uint16, sparse=True) 
        row_labels = list(tsMatrix) 
        trainset = vectorizer.fit_transform([tsMatrix[i] for i in row_labels]).T
        trainset_created.append(trainset)

        tsMatrix_test = algorithm.test(test_dataset)
        row_labels = list(tsMatrix_test) 
        testset = vectorizer.fit_transform([tsMatrix_test[i] for i in row_labels]).T
        testset_created.append(testset)

    trainset = hstack(trainset_created)
    testset = hstack(testset_created)

    scaler = StandardScaler(with_mean = False)
    
    if classifier == 'Ridge':
        cls = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
    elif classifier == 'Elastic':
        cls = SGDClassifier(max_iter=1000, tol=1e-3, penalty='elasticnet')
    elif classifier == 'Lasso':
        cls = SGDClassifier(max_iter=1000, tol=1e-3, penalty='l1')
    else:
        cls = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
    
    trainset = scaler.fit_transform(trainset)
    cls.fit(trainset, y_train)

    lr_score = cls.score(testset, y_test)
    t6 = time.perf_counter()
    timegap = t6-t1

    return lr_score, timegap
    

def simpleTrial(data, alphabet_size = 5, alphabet_size_slope = 5, 
                window_size = 10, split_no = 1, step=10, 
                multivariate=True, varyingLength=False,
                classifier = 'Ridge'):
    X_train = znorm(data['TRAIN']['X'])
    y_train = data["TRAIN"]["y"]
    X_test = znorm(data['TEST']['X'])
    y_test = data["TEST"]["y"]

    if multivariate == True:
        lr_score, timegap = trialSequence_multivariate(X_train, X_test, y_train, y_test, window_size = window_size, 
            window_size_slope = 1, alphabet_size = alphabet_size, alphabet_size_slope=alphabet_size_slope, step=step, split_no = split_no, 
            varyingLength=varyingLength, classifier = classifier)
    else:    
        lr_score, timegap = trialSequence(X_train, X_test, y_train, y_test, window_size = window_size, 
            window_size_slope = 1, alphabet_size = alphabet_size, alphabet_size_slope=alphabet_size_slope,split_no = split_no, step=step, 
            classifier = classifier)

    return lr_score, timegap

def randomizedSearch(data, steps = [10, 20, 30, 50, 100], 
                        alphabet_sizes = [3, 5, 7, 10], 
                        alphabet_sizes_slope = [3, 5, 7, 10], 
                        window_sizes = [3, 5, 7, 10], 
                        split_nos = [1, 2, 4, 8], cycle = 5, 
                        multivariate=True, varyingLength=False,
                        classifier = 'Ridge'):
    if varyingLength == False: 
        X_train = znorm(data['TRAIN']['X'])
        X_test = znorm(data['TEST']['X'])
    else:
        X_train = [znorm(np.array([i.values for i in data["TRAIN"]["X"].iloc[j]])) for j in range(data["TRAIN"]["X"].shape[0])]
        X_test = [znorm(np.array([i.values for i in data["TEST"]["X"].iloc[j]])) for j in range(data["TEST"]["X"].shape[0])]

    y_train = data["TRAIN"]["y"]
    y_test = data["TEST"]["y"]

    best_params = []
    best_acc = -1
    lr_score = 0

    try:
        X_train_rs, X_val_rs, y_train_rs, y_val_rs = train_test_split(X_train, y_train, test_size=0.33, stratify=y_train)
    except:
        X_train_rs, X_val_rs, y_train_rs, y_val_rs = train_test_split(X_train, y_train, test_size=0.33)

    count = 0
    
    while True:
        step = np.random.choice(steps, 1)[0]
        alphabet_size = np.random.choice(alphabet_sizes, 1)[0]
        alphabet_size_slope = np.random.choice(alphabet_sizes_slope, 1)[0]
        window_size = np.random.choice(window_sizes, 1)[0]
        split_no = np.random.choice(split_nos, 1)[0]

        try:
            if multivariate:
                lr_score, timegap = trialSequence_multivariate(X_train_rs, X_val_rs, y_train_rs, y_val_rs, window_size = window_size, 
                            window_size_slope = 1, alphabet_size = alphabet_size, alphabet_size_slope=alphabet_size_slope, step=step, split_no = split_no, 
                            varyingLength=varyingLength, classifier=classifier)
            else:
                lr_score, timegap = trialSequence(X_train_rs, X_val_rs, y_train_rs, y_val_rs, window_size = window_size, 
                            window_size_slope = 1, alphabet_size = alphabet_size, alphabet_size_slope=alphabet_size_slope, step=step, split_no = split_no,
                            classifier=classifier)
        except:
            print("TrialSequence Failed")
            continue

        if lr_score == None:
            continue

        if lr_score > best_acc:
            best_acc = lr_score
            best_params = [window_size, 1, alphabet_size, alphabet_size_slope, split_no, step]
            print("BEST PARAM CHANGED: ", count, best_acc, best_params)
        count += 1

        if count == cycle:
            break

    if multivariate:
        lr_score, timegap = trialSequence_multivariate(X_train, X_test, y_train, y_test, window_size = best_params[0], 
            window_size_slope = best_params[1], alphabet_size = best_params[2], alphabet_size_slope=best_params[3], split_no = best_params[4], step=best_params[5], varyingLength=varyingLength,
            classifier = classifier)
    else:
        lr_score, timegap = trialSequence(X_train, X_test, y_train, y_test, window_size = best_params[0], 
            window_size_slope = best_params[1], alphabet_size = best_params[2], alphabet_size_slope=best_params[3], split_no = best_params[4], step=best_params[5],
            classifier = classifier)

    return lr_score, timegap, best_params
