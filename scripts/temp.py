"""
temporary script for tests
"""
from scipy import sparse

from random import sample

dtm = sparse.load_npz('data/data_preprocess/dtm_occ.npz')

sparse.save_npz('tests/dtm_test.npz', dtm)

dtm = sparse.load_npz('tests/dtm_test.npz')