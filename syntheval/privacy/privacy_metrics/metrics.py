import re
import pickle
import pandas as pd
from typing import Union

def entity_leakage(paragraphs: list, entities: list, entity_leakage_result_path: str) -> Union[float, dict]:
    """
    Check if each entity is present in the given paragraphs.

    Args:
        paragraphs (list): A list of paragraphs to search in. Normally, it is the output of a model.
        entities (list): A list of entities (strings) to search for.
        entity_leakage_result_path (str): The path to save the results.
        
    Returns:
        float: The overall percentage of entities leaked in the paragraphs
        dict: A dictionary where each key is a paragraph and the value is a dictionary of entities and their presence.
    """
    results = {}
    total_leaked_count, total_entities = 0, len(entities)
    for paragraph in paragraphs:
        result, leaked_count = entity_leakage_per_paragraph(paragraph, entities)
        total_leaked_count+=leaked_count
        results[paragraph] = (result, leaked_count)
        
    # Save results into a pickle file
    if(entity_leakage_result_path is not None):
        with open(entity_leakage_result_path, "wb") as f:
            pickle.dump(results, f)

    return (total_leaked_count / (total_entities * len(paragraphs))) * 100, results

def entity_leakage_per_paragraph(paragraph: str, entities: list) -> dict:
    """
    Check if each entity is present in the given paragraph.

    Args:
        paragraph (str): The text to search in. Normally, it is the output of a model.
        entities (list): A list of entities (strings) to search for.

    Returns:
        dict: A dictionary where each key is an entity and the value is True if found, else False.
    """
    results = {}
    leaked_count, total_entities = 0, len(entities)
    for entity in entities:
        cleaned_entity = entity.strip('"')
        pattern = r'\b' + re.escape(cleaned_entity) + r'\b'
        found = re.search(pattern, paragraph, re.IGNORECASE) is not None
        results[cleaned_entity] = found
        if(found):
            leaked_count+=1
    return results, (leaked_count / total_entities) * 100

def phrase_search(documents, patterns, window = 3):
    
    entity_phrase_spans, window_lengths, entities = [], [], []
    for doc in documents:
        for pattern in patterns:
            escaped_pattern = re.escape(pattern)
            # Build the dynamic regex pattern
            context_pattern = re.compile(fr'((?:\S+\s+){{0,{window}}}){escaped_pattern}((?:\s+\S+){{0,{window}}})')
            matches = context_pattern.finditer(doc)
            for match in matches:
                pattern_in_context = match.group(0).strip()
                #combined_match = f"{pattern}".strip()
                words = pattern_in_context.split()
                n = len(words)
                for w in range(window):
                    combined_match = (" ".join(words[w:n-w])).strip()
                    if(pattern in combined_match):
                        entity_phrase_spans.append(combined_match)
                        window_lengths.append(int(window - w))
                        entities.append(pattern)
            
    return entity_phrase_spans, window_lengths, entities

def search_phrase(df, patterns, save_file_path = 'outputs.csv', max_window_len = 4, text_field = "output_text"):
    
    try:
        df = pd.read_csv(df)
    except:
        df = df
    
    print("Length:", len(df))
    print("Total number of entities", len(patterns))

    phrases, window_lengths, entities = phrase_search(df[text_field].tolist(), patterns, window = max_window_len)
    
    df = pd.DataFrame({'Entity': entities, 'Phrase': phrases, 'Context Length': window_lengths})
    df.to_csv(save_file_path)
    return phrases
