import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# read the handled data
X_train=pd.read_csv('train_features')
X_test = pd.read_csv('test_features')
y = pd.read_csv('train_label',header=None) # actually i didnt store the header for the labels
