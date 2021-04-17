"""
    Create Document-Thematics matrix and proceed AFC on it
"""
import pickle

from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import FactorAnalysis

if __name__ == '__main__':

    # -------------------------
    # Thematics-documents matrix
    # -------------------------
      
    with open('../data/data_preprocess/subforums.pkl', 'rb') as file:
        subforums = pickle.load(file)

    data = [item[0] for item in list(subforums.questions.index)]
    onehot_encoder = OneHotEncoder(sparse=True)
    # for questions
    P_q = onehot_encoder.fit_transform(data.reshape(len(data), 1))

    data = [item[0] for item in list(subforums.answers.index)]
    onehot_encoder = OneHotEncoder(sparse=True)
    # for answers
    P_a = onehot_encoder.fit_transform(data.reshape(len(data), 1))
    
    # -------------------------
    # Thematics-terms matrix
    # -------------------------

    with open('../data/data_preprocess/questions_DTM_occ.pkl', 'rb') as file:
        Q_count = pickle.load(file)
    with open('../data/data_preprocess/answers_DTM_occ.pkl', 'rb') as file:
        A_count = pickle.load(file)

    TT_q_occ = P_q.transpose().multiply(Q_count)
    TT_a_occ = P_a.transpose().multiply(A_count)

    with open('../data/data_preprocess/questions_DTM_tfidf.pkl', 'rb') as file:
        Q_tfidf = pickle.load(file)
    with open('../data/data_preprocess/answers_DTM_tfidf.pkl','rb') as file:
        A_tfidf = pickle.load(file)

    TT_q_tfidf = P_q.transpose().multiply(Q_tfidf)
    TT_a_tfidf = P_a.transpose().multiply(A_tfidf)

    with open('../data/thematics_terms/questions_TTM_occ.pkl', 'wb') as file:
        pickle.dump(TT_q_occ, file)
    
   
    with open('../data/thematics_terms/answers_TTM_occ.pkl', 'wb') as file:
        pickle.dump(TT_a_occ, file)
    with open('../data/thematics_terms/questions_TTM_tfidf.pkl', 'wb') as file:
        pickle.dump(TT_q_tfidf, file)
    with open('../data/thematics_terms/answers_TTM_tfidf.pkl', 'wb') as file:
        pickle.dump(TT_a_tfidf, file)
    

    # -------------------------
    # AFC
    # -------------------------

    #apply AFC to any thematic_term_matrix obtained before
        
    # apply AFC to TT_q_occ matrix
    transformer = FactorAnalysis(n_components=7, random_state=0)  #choix de n_comp ?
    TT_q_occ_transformed = transformer.fit_transform(TT_q_occ)
    TT_q_occ_transformed.shape
    
    # apply AFC to TT_a_occ matrix
    transformer = FactorAnalysis(n_components=7, random_state=0)  
    TT_a_occ_transformed = transformer.fit_transform(TT_a_occ)
    TT_a_occ_transformed.shape
    
    # apply AFC to TT_q_tfidf matrix
    transformer = FactorAnalysis(n_components=7, random_state=0)  
    TT_q_tfidf_transformed = transformer.fit_transform(TT_q_tfidf)
    TT_q_tfidf_transformed.shape
    
    # apply AFC to TT_a_tfidf matrix
    transformer = FactorAnalysis(n_components=7, random_state=0)  
    TT_a_tfidf_transformed = transformer.fit_transform(TT_a_tfidf)
    TT_a_tfidf_transformed.shape 