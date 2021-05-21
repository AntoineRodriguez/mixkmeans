from scipy import sparse
import numpy as np
import random

import pickle

with open('notebooks/occ-cosin_save-100-567.pkl', 'rb') as file:
    objet = pickle.load(file)