import nltk
import spacy
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from synthtexteval.eval.descriptive.arguments import TextDescriptorArgs
from synthtexteval.eval.descriptive.compare import basic_comparison_metrics, compare_distributions
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
from gensim.parsing.preprocessing import preprocess_documents
from itertools import chain
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

nltk.download('punkt')
nltk.download('punkt_tab')

class TextDescriptor:
    """
    Defines the text descriptor class for analyzing textual data. The class provides methods for analyzing named entities,
    n-grams, TF-IDF scores, and topic modeling. The results can be saved as a pickle file and optionally plotted.
    Attributes:
        texts (list): A list of textual documents.
        args (TextDescriptorArgs): Configuration arguments for text processing and analysis.
    """
    def __init__(self, texts, args: TextDescriptorArgs, reference_texts = []):
        self.texts = texts
        self.reference_texts = reference_texts
        self.tokenized_texts = [nltk.word_tokenize(doc.lower()) for doc in self.texts]
        self.nlp = spacy.load("en_core_web_sm")
        self.entity_counter = self._get_entity_count()
        self.args = args
    
    def _get_entity_count(self):
        """
        Get the count of all the entities in the document.
        """
        entity_counter = Counter()
        for text in self.texts:
            doc = self.nlp(text)
            for ent in doc.ents:
                entity_counter[ent.text] += 1
        self.entity_counter = entity_counter

    def _get_least_frequent_entities(self, n):
        """
        Returns a tuple of the least frequent entities in the document and their corresponding frequency.
        """
        sorted_entities = sorted(self.entity_counter.items(), key=lambda x: x[1])
        min_count = sorted_entities[n-1][1] if len(sorted_entities) >= n else sorted_entities[-1][1]
        return {key: value for key, value in self.entity_counter.items() if value <= min_count}
    
    def _preprocess_for_topic_modeling(self):
        """
        Preprocess the text for topic modeling with Gensim.
        """
        self.tokenized_texts_lda = preprocess_documents([' '.join(i) for i in self.tokenized_texts])
    
    def _get_top_n_entities(self, top_n):
        """
        Returns a tuple of the most frequent entities in the document and their corresponding frequency.
        """
        return sorted(self.entity_counter.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def _plot_entity_frequency(self, plt_file_path):
        """
        Plots the most frequent entities by their frequency.
        """
        sorted_entities = sorted(self.entity_counter.items(), key=lambda x: x[1], reverse=True)
        entities, counts = zip(*sorted_entities[:20])
        
        plt.figure(figsize=(10, 6))
        plt.barh(entities, counts, color='skyblue')
        plt.xlabel('Frequency')
        plt.ylabel('Entity')
        plt.title('Top 20 Entities Frequency Distribution')
        plt.gca().invert_yaxis()
        plt.savefig(plt_file_path)
    
    def _ngram_frequency(self, n_gram_size=2, top_k=5):
        """
        Find the most and least common n-grams.
        
        Arguments:
            (int) n_gram_size: N-gram size
            (int) top_k: Number of top/bottom n-grams to return
        
        Returns:
            tuple (str, int): Most frequent n-grams of size n_gram_size
            tuple (str, int): Least frequent n-grams of size n_gram_size
        """
        ngrams = list(chain.from_iterable([list(nltk.ngrams(doc, n_gram_size)) for doc in self.tokenized_texts]))
        ngram_counts = Counter(ngrams)
        most_common = ngram_counts.most_common(top_k)
        least_common = ngram_counts.most_common()[:-top_k-1:-1]
        return most_common, least_common
    
    def _compute_tfidf(self):
        """
        Compute TF-IDF scores for the documents.

        Returns: 
            (str): Feature names
            (np.array): TF-IDF matrix
        """
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.texts)
        return vectorizer.get_feature_names_out(), tfidf_matrix.toarray()
    
    def _topic_modeling(self, num_topics=3, num_words=5, display=True):
        """Perform LDA topic modeling.
        
        Arguments:
            (int) num_topics: Number of topics
            (int) num_words: Number of words per topic
        
        Returns: 
            (list): List of topics with words
        """
        self._preprocess_for_topic_modeling()
        dictionary = corpora.Dictionary(self.tokenized_texts_lda)
        corpus = [dictionary.doc2bow(text) for text in self.tokenized_texts_lda]
        lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
        topics = lda_model.print_topics(num_words=num_words)
        return topics
    
    def _topic_modeling_display(self, num_topics=3):
        """Perform LDA topic modeling, and displays result wiht pyLDAvis
        
        Arguments:
            (int) num_topics: Number of topics
        Returns: 
            (list): LDA display
        """
        self._preprocess_for_topic_modeling()
        dictionary = corpora.Dictionary(self.tokenized_texts_lda)
        corpus = [dictionary.doc2bow(text) for text in self.tokenized_texts_lda]
        lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
        return gensimvis.prepare(lda_model, corpus, dictionary)
            
    def _compare_to_reference_distribution(self, metrics, plot=False):
        """
        Compare the text distributions based on the provided metrics.
        
        Arguments:
            (list) metrics: List of metrics to compare
        """
        if 'text-length' in metrics:
            print("Comparing text length...")
            basic_comparison_metrics(self.texts, self.reference_texts, plot)
        
        print("Comparing distributions...")
        compare_distributions(self.texts, self.reference_texts, metrics)

    def analyze_entities(self):
        """
        Analyzes named entities by identifying the most and least frequent ones, saves the results as a 
        pickle file, and optionally generates a plot of entity frequencies.
        """
        least_frequent = self._get_least_frequent_entities(n = self.args.min_threshold)
        if len(least_frequent) > self.args.min_threshold:
            least_frequent_keys = list(least_frequent.keys())
            least_frequent = {least_frequent_keys[i]: least_frequent[least_frequent_keys[i]] for i in range(self.args.min_threshold)}
        most_frequent = self._get_top_n_entities(top_n = self.args.max_threshold)
        
        print("Most frequent entities:", most_frequent)
        print("Least frequent entities:", least_frequent)
        print("Saving the pickle results to:", self.args.pkl_file_path)
        
        self.save_to_pickle({'entity_count': self.entity_counter, 'least_frequent': least_frequent, 'most_frequent': most_frequent}, pkl_file_path = self.args.pkl_file_path)
        
        if self.args.produce_plot:
            print("Saving the plot figure to: ", self.args.plt_file_path)
            self._plot_entity_frequency(plt_file_path = self.args.plt_file_path)
    
    def save_to_pickle(self, data, pkl_file_path):
        """
        Save the analysis results to a pickle file.
        """
        with open(pkl_file_path, 'wb') as file:
            pickle.dump(data, file)