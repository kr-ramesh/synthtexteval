import numpy as np
from scipy.special import kl_div
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def convert_to_tfidf(texts_1, texts_2):
    """
    Convert texts to TF-IDF vectors.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix_1 = vectorizer.fit_transform(texts_1).toarray()
    tfidf_matrix_2 = vectorizer.transform(texts_2).toarray()
    
    return tfidf_matrix_1, tfidf_matrix_2

def kl_divergence(texts_1, texts_2):
    """
    Calculate the Kullback-Leibler divergence between two sets of texts.
    """

    tfidf_matrix_1, tfidf_matrix_2 = convert_to_tfidf(texts_1, texts_2)
    kl_divs = []
    for doc_1 in tfidf_matrix_1:
        for doc_2 in tfidf_matrix_2:
            kl_divs.append(np.sum(kl_div(doc_1, doc_2)))

    return np.mean(kl_divs)

def jaccard_similarity(texts_1, texts_2):
    """
    Calculate the Jaccard similarity between two sets of texts.
    """
    
    def jaccard(set1, set2):
        return len(set1.intersection(set2)) / len(set1.union(set2))

    similarities = []
    for t1 in texts_1:
        set1 = set(t1.split())
        for t2 in texts_2:
            set2 = set(t2.split())
            similarities.append(jaccard(set1, set2))

    return np.mean(similarities)

def cosine_similarity_between_texts(texts_1, texts_2):
    """
    Calculate the cosine similarity between two sets of texts.
    """

    tfidf_matrix_1, tfidf_matrix_2 = convert_to_tfidf(texts_1, texts_2)
    cosine_sim = cosine_similarity(tfidf_matrix_1, tfidf_matrix_2)

    return np.mean(np.mean(cosine_sim, axis = 1))