"""
    Create Document-Thematics matrix and proceed AFC on it
"""
import pickle

import numpy as np
from scipy import sparse
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # -------------------------
    # questions/answers indexes
    # -------------------------
    
    #read csv
    
    subforum_answers = pd.read_csv('./data/data_preprocess/subforum_answers.csv')
    subforum_questions= pd.read_csv('./data/data_preprocess/subforum_questions.csv')
    
    index_answers = list(subforum_answers.iloc[:,0])
    index_questions = list(subforum_questions.iloc[:,0])

    data = [item[0] for item in index_questions]
    onehot_encoder = OneHotEncoder(sparse=True)
    # for questions
    P_q = onehot_encoder.fit_transform(np.array(data).reshape(-1, 1))

    data = [item[0] for item in index_answers]
    onehot_encoder = OneHotEncoder(sparse=True)
    # for answers
    P_a = onehot_encoder.fit_transform(np.array(data).reshape(-1, 1))
    
    # -------------------------
    # Thematics-terms matrix
    # -------------------------

    A_count = sparse.load_npz('./data/data_preprocess/dtm_answers_occ.npz')
    Q_count = sparse.load_npz('./data/data_preprocess/dtm_questions_occ.npz')

    TT_q_occ = P_q.transpose().dot(Q_count)
    TT_a_occ = P_a.transpose().dot(A_count)

    A_tfidf = sparse.load_npz('./data/data_preprocess/dtm_answers_tfidf.npz')
    Q_tfidf = sparse.load_npz('./data/data_preprocess/dtm_questions_tfidf.npz')

    TT_q_tfidf = P_q.transpose().dot(Q_tfidf)
    TT_a_tfidf = P_a.transpose().dot(A_tfidf)


    #####
    #save TT Matrix
    ####
    with open('./data/thematics_terms/TTM_questions_occ.pkl', 'wb') as file:
        pickle.dump(TT_q_occ, file)   
    with open('./data/thematics_terms/TTM_answers_occ.pkl', 'wb') as file:
        pickle.dump(TT_a_occ, file)
    with open('./data/thematics_terms/TTM_questions_tfidf.pkl', 'wb') as file:
        pickle.dump(TT_q_tfidf, file)
    with open('./data/thematics_terms/TTM_answers_tfidf.pkl', 'wb') as file:
        pickle.dump(TT_a_tfidf, file)
    

    # -------------------------
    # AFC
    # -------------------------

    #apply AFC to any thematic_term_matrix obtained before
        
    # apply AFC to TT_q_occ matrix
    transformer = FactorAnalysis()  
    TT_q_occ_transformed = transformer.fit_transform(TT_q_occ.transpose().toarray())
    TT_q_occ_transformed.transpose().shape  #matrix 4*m
    
    # apply AFC to TT_a_occ matrix
    transformer = FactorAnalysis()  
    TT_a_occ_transformed = transformer.fit_transform(TT_a_occ.transpose().toarray())
    TT_a_occ_transformed.transpose().shape
   
    # apply AFC to TT_q_tfidf matrix
    transformer = FactorAnalysis()  
    TT_q_tfidf_transformed = transformer.fit_transform(TT_q_tfidf.transpose().toarray())
    TT_q_tfidf_transformed.transpose().shape
    
    # apply AFC to TT_a_tfidf matrix
    transformer = FactorAnalysis()  
    TT_a_tfidf_transformed = transformer.fit_transform(TT_a_tfidf.transpose().toarray())
    TT_a_tfidf_transformed.transpose().shape

# TODO: plot sur le 1er plan factoriel
    
    
######################################## method test    
  
from factor_analyzer import FactorAnalyzer

fa = FactorAnalyzer()
f = fa.fit_transform(TT_q_occ.transpose().toarray())
f.transpose().shape  #------ #3*m
    
# Create scree plot using matplotlib
plt.scatter(range(1,TT_q_occ.transpose().shape[1]+1),ev)
plt.plot(range(1,TT_q_occ.transpose().shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

#from fanalysis.ca import CA