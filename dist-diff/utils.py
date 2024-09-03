from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
import numpy as np

def jaccard_similarity(corpus1, corpus2):
    """
    # Example usage
        corpus1 = ["This is a document.", "Another document here."]
        corpus2 = ["This document is different.", "Another one."]
        print("Jaccard Similarity:", jaccard_similarity(corpus1, corpus2))
    """
    
    vectorizer = CountVectorizer(binary=True)
    X1 = vectorizer.fit_transform(corpus1).toarray()
    X2 = vectorizer.transform(corpus2).toarray()
    
    # Compute Jaccard similarity between all pairs of documents
    similarities = []
    for i in range(X1.shape[0]):
        for j in range(X2.shape[0]):
            similarity = jaccard_score(X1[i], X2[j])
            similarities.append(similarity)
    
    return np.mean(similarities)

def lda_similarity(corpus1, corpus2, n_topics=5):
    """
    # Example usage
        corpus1 = ["This is a document about machine learning.", "Another document about AI."]
        corpus2 = ["This document discusses algorithms.", "Another text about learning algorithms."]
        print("LDA KL Divergence:", lda_similarity(corpus1, corpus2))
    
    """

    vectorizer = CountVectorizer()
    X1 = vectorizer.fit_transform(corpus1)
    X2 = vectorizer.transform(corpus2)
    
    lda = LatentDirichletAllocation(n_components=n_topics)
    lda.fit(X1)
    topic_distribution1 = lda.transform(X1).mean(axis=0)
    lda.fit(X2)
    topic_distribution2 = lda.transform(X2).mean(axis=0)
    
    # Compute KL divergence
    def kl_divergence(p, q):
        return np.sum(p * np.log(p / q + 1e-10))
    
    kl_div = kl_divergence(topic_distribution1, topic_distribution2)
    
    return kl_div
