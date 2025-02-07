import spacy
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from collections import Counter

#TODO: Need to add more descriptive measures - what other functions can we add to this?

nlp = spacy.load('en_core_web_sm')

def get_entity_count(texts):
    entity_counter = Counter()
    for text in texts:
        doc = nlp(text)
        for ent in doc.ents:
            entity_counter[ent.text] += 1
    return entity_counter

def get_least_frequent_entities(entity_counter, threshold=1):
    least_frequent = {key: value for key, value in entity_counter.items() if value <= threshold}
    return least_frequent

def plot_entity_frequency(entity_counter, plt_file_path):

    sorted_entities = sorted(entity_counter.items(), key=lambda x: x[1], reverse=True)
    entities, counts = zip(*sorted_entities)
    
    # Plot the frequency distribution
    plt.figure(figsize=(10, 6))
    plt.barh(entities[:20], counts[:20], color='skyblue')
    plt.xlabel('Frequency')
    plt.ylabel('Entity')
    plt.title('Top 20 Entities Frequency Distribution')
    plt.gca().invert_yaxis()
    plt.save_fig(plt_file_path)

def save_to_pickle(data, filename='entity_data.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def analyze_texts(df, text_column, produce_plot = False):

    texts = df[text_column].tolist()
    entity_counter = get_entity_count(texts)

    least_frequent = get_least_frequent_entities(entity_counter)
    
    #TODO: Change this
    if(produce_plot):
        plot_entity_frequency(entity_counter, "plot.png")
    
    save_to_pickle({'entity_count': entity_counter, 'least_frequent': least_frequent}, 'entity_analysis.pkl')


"""
#Example usage
data = {
    'texts': [
        "Apple is looking at buying U.K. startup for $1 billion",
        "Microsoft has acquired GitHub for $7.5 billion",
        "Amazon's new headquarters will be in Virginia",
        "Tesla's new electric car is amazing"
    ]
}

df = pd.DataFrame(data)
analyze_texts(df, 'texts')"""
