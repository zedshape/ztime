from ztime import helper
import pickle
import warnings
warnings.filterwarnings('ignore')

with open('data/BasicMotions.pickle', 'rb') as handle:
    data = pickle.load(handle)

score, time = helper.simpleTrial(data)
print(f"score: {score}, time: {time}")