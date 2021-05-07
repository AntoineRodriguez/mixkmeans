import pickle

from scipy import sparse

from mixkmeans import MixKMeans

if __name__ == '__main__':
    dtm = sparse.load_npz('data/data_preprocess/dtm_occ.npz')

    model = MixKMeans(x=-3, weights=(0.2, 0.8))

    # TODO: Training / Test Split
    prototypes, assignation, cost = model.fit(dtm, K=4, itermax=100)
    # no needs to fit

    # save results
    with open('data/training_results/results_1.pkl', 'wb') as file:
        pickle.dump([prototypes, assignation, cost], file)
