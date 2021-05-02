"""
temporary script for tests
"""
from scipy import sparse

from random import sample

dtm = sparse.load_npz('data/data_preprocess/dtm_occ.npz')

sparse.save_npz('tests/dtm_test.npz', dtm)

dtm = sparse.load_npz('tests/dtm_test.npz')

iter = 0
for i in dtm:
    print(i)
    iter += 1
    print(10*'---')
    if iter == 3:
        break