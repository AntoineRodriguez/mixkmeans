from scipy import sparse
import numpy as np
import random

mat = sparse.load_npz('./data/data_preprocess/dtm_occ.npz')

liste = random.sample(range(21939), 200)

sparse.save_npz('./data/data_preprocess/dtm_test.npz', mat[liste])


mat = sparse.load_npz('tests/dtm_tes')

for i in range(5):
    with open('bidule.txt', 'w') as file:
        file.write(str(i))

file = open('bidule.txt', 'w')
for item in range(5):
    if item == 3:
        raise ValueError
    file.write(str(item)+'\n')
file.close()


### OPEN NPZ
a = np.load('notebooks/backup/tfidf_eucl_save-100.npz')
a['arr_0']
a['arr_1']