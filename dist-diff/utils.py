from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import jaccard_score
from scipy.stats import entropy
import numpy as np
import pandas as pd

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

def additional_metrics(real_texts, synthetic_texts):

    all_texts = real_texts + synthetic_texts

    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Split back into real and synthetic matrices
    tfidf_real = tfidf_matrix[:len(real_texts)]
    tfidf_synthetic = tfidf_matrix[len(real_texts):]

    # Compute the average cosine similarity between real and synthetic distributions
    cos_sim_matrix = cosine_similarity(tfidf_real, tfidf_synthetic)
    average_cos_sim = np.mean(cos_sim_matrix)

    print(f"Average Cosine Similarity: {average_cos_sim:.4f}")

    # Calculate the KL divergence between the average distributions
    real_distribution = np.mean(tfidf_real.toarray(), axis=0)
    synthetic_distribution = np.mean(tfidf_synthetic.toarray(), axis=0)

    # Adding a small constant to avoid zero issues
    kl_divergence = entropy(real_distribution + 1e-10, synthetic_distribution + 1e-10)

    print(f"KL Divergence: {kl_divergence:.4f}")

    # Jensen-Shannon divergence between the distributions
    js_divergence = jensenshannon(real_distribution, synthetic_distribution, base=2)

    print(f"Jensen-Shannon Divergence: {js_divergence:.4f}")

    return average_cos_sim, kl_divergence, js_divergence
