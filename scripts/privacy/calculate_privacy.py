import re
import sys
import pandas as pd
from datasets import load_from_disk
from synthtexteval.eval.privacy.metrics import entity_leakage, search_and_compute_EPO

real_texts = load_from_disk(sys.argv[2])
real_texts = real_texts['train']
synth_df = pd.read_csv(sys.argv[1])

entities = []
for i in real_texts['annotations']:
    try:
        for annotator in i:
            for entity in i[annotator]['entity_mentions']:
                if(entity['entity_type'] in ['PERSON', 'DATETIME']):
                    entities.append(entity['span_text'])
    except Exception as e:
        continue

# Comptues the percentage of leaked entities
total_leakage, privacy_analysis = entity_leakage(synth_df['output_text'].tolist(), entities, 'privacy-leakage.pkl')
print(f"Percentage of leaked entities: {100*total_leakage:.3f} %")

text_field = 'output_text'
# Computes the entity context memorization statistics
search_and_compute_EPO(synth_file = synth_df, reference_texts = real_texts['train']['text'], 
                       synth_phrase_file_path = 'synth-outputs.csv',
                       entity_patterns = entities, max_window_len = 3,
                       text_field = text_field)