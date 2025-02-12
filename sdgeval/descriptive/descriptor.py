import spacy
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from sdgeval.descriptive.arguments import DescriptorArgs


class Descriptor:
    def __init__(self, texts, args: DescriptorArgs):
        self.texts = texts
        self.nlp = spacy.load("en_core_web_sm")
        self.entity_counter = self._get_entity_count()
        self.args = args
    
    def _get_entity_count(self):
        entity_counter = Counter()
        for text in self.texts:
            doc = self.nlp(text)
            for ent in doc.ents:
                entity_counter[ent.text] += 1
        return entity_counter
    
    def save_to_pickle(self, data, pkl_file_path):
        with open(pkl_file_path, 'wb') as file:
            pickle.dump(data, file)

    def get_least_frequent_entities(self, n):
        sorted_entities = sorted(self.entity_counter.items(), key=lambda x: x[1])
        min_count = sorted_entities[n-1][1] if len(sorted_entities) >= n else sorted_entities[-1][1]
        return {key: value for key, value in self.entity_counter.items() if value <= min_count}
    
    def get_top_n_entities(self, top_n):
        return sorted(self.entity_counter.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def plot_entity_frequency(self, plt_file_path):
        sorted_entities = sorted(self.entity_counter.items(), key=lambda x: x[1], reverse=True)
        entities, counts = zip(*sorted_entities[:20])
        
        plt.figure(figsize=(10, 6))
        plt.barh(entities, counts, color='skyblue')
        plt.xlabel('Frequency')
        plt.ylabel('Entity')
        plt.title('Top 20 Entities Frequency Distribution')
        plt.gca().invert_yaxis()
        plt.savefig(plt_file_path)
            
    def analyze(self):
        least_frequent = self.get_least_frequent_entities(n = self.args.min_threshold)
        most_frequent = self.get_top_n_entities(top_n = self.args.max_threshold)
        
        print("Most frequent entities:", most_frequent)
        print("Least frequent entities:", least_frequent)
        print("Saving the pickle results to:", self.args.pkl_file_path)
        
        self.save_to_pickle({'entity_count': self.entity_counter, 'least_frequent': least_frequent, 'most_frequent': most_frequent}, pkl_file_path = self.args.pkl_file_path)
        
        if self.args.produce_plot:
            print("Saving the plot figure to: ", self.args.plt_file_path)
            self.plot_entity_frequency(plt_file_path = self.args.plt_file_path)
