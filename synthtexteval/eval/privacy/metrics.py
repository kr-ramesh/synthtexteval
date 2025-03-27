import re
import pickle
import pandas as pd
from tabulate import tabulate
from typing import Union

def entity_leakage(paragraphs: list, entities: list, entity_leakage_result_path: str) -> Union[float, dict]:
    """
    Check if each entity is present in the given paragraphs.

    Args:
        paragraphs (list): A list of paragraphs to search in. Normally, it is the output of a model.
        entities (list): A list of entities (strings) to search for.
        entity_leakage_result_path (str): The path to save the results.
        
    Returns:
        float: The average number of entities leaked per paragraph
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

    return (total_leaked_count / len(paragraphs)), results

def search_and_compute_EPO(synth_file, reference_texts, synth_phrase_file_path, entity_patterns, max_window_len=4, 
                           remove_duplicates=True, text_field = 'output_text'):

    """
    Searches and computes the Entity Phrase Overlap (EPO).
    The Entity Phrase Overlap is a metric that quantifies the overlap of phrases containing specific entities
    between a synthetic file and a reference file.
    We define a context window around the matched entities to capture the surrounding text, which helps in understanding
    the context in which the entities are mentioned and memorized.

    Args:
        synth_file (str): Path or Pandas DataFrame to the synthetic file containing text 
        ref_file (str): Path to Pandas DataFrame or HF Dataset to the reference file containing text.
        synth_phrase_file_path (str): Path to save the synthetic file with matched phrases and context lengths.
    
    Returns:
        pd.DataFrame: A DataFrame containing the overlap count and ratio for each context length.
    
    """

    # Search for phrases in the synthetic file
    df = search_phrase_text(synth_file, entity_patterns, save_file_path=synth_phrase_file_path, max_window_len=max_window_len, text_field=text_field)
    
    # Compute the overlap between the synthetic and reference files
    overlap_df = compute_phrase_text_overlap(synth_phrase_file_path, reference_texts, remove_duplicates=remove_duplicates)

    return overlap_df


# Helper functions

def entity_leakage_per_paragraph(paragraph: str, entities: list) -> dict:
    """
    Check if each entity is present in the given paragraph.

    Args:
        paragraph (str): The text to search in. Normally, it is the output of a model.
        entities (list): A list of entities (strings) to search for.

    Returns:
        dict: A dictionary where each key is an entity and the value is True if found, else False.
        leaked_count : The number of unique entities from the training leaked in the synthetic text
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
    return results, leaked_count

def context_pattern_match(documents, patterns, window = 3):

    """
    Search for phrases in the provided documents that match the entities/patterns, 
    allowing for a specified context window around the matched entities.
    """
    
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

def search_phrase_text(df, patterns, save_file_path = 'outputs.csv', max_window_len = 4, text_field = "output_text"):
    
    """
    This function reads a CSV file or DataFrame, searches for the specified entity patterns in the text,
    and saves the matched phrases along with their context lengths to a new CSV file.

    Args:
        df (str or pd.DataFrame): Path to the CSV file or a DataFrame containing the documents.
        patterns (list): List of entities/patterns to search for in the documents.
        save_file_path (str): Path to save the output CSV file with matched phrases and their context lengths.
        max_window_len (int): Maximum number of words to include in the context window around the matched entities.
        text_field (str): The column name in the DataFrame that contains the text to search.    
    
    """

    try:
        df = pd.read_csv(df)
    except:
        df = df
    
    print("Length:", len(df))
    print("Total number of entities", len(patterns))

    phrases, window_lengths, entities = context_pattern_match(df[text_field].tolist(), patterns, window = max_window_len)
    
    df = pd.DataFrame({'Entity': entities, 'Phrase': phrases, 'Context Length': window_lengths})
    df.to_csv(save_file_path, index = False)
    print(f"Output saved to {save_file_path}")

    return df
    
def compute_phrase_text_overlap(synth_file_path, reference_texts, remove_duplicates=True):

    """
    This function compares the phrases extracted from a synthetic file with those from a reference file,
    and returns the count of phrases of a given context window that appear divided by the total number of training points.

    Args:
        synth_file_path (str): Path to the synthetic file containing phrases.
        reference_texts (lst): List of texts in the training data
        remove_duplicates (bool): Flag to indicate whether to remove duplicates from the dataframes before comparison.
    
    Returns:
        pd.DataFrame: A DataFrame containing the overlap count and ratio for each context length.
    """
    match_found = []
    df = pd.read_csv(synth_file_path)  
    df['Context Length'] = df['Context Length'].astype(int)
    if(remove_duplicates):
        df.drop_duplicates(subset=['Entity', 'Phrase', 'Context Length'], inplace=True)

    for phrase in df['Phrase'].tolist():
        if any(phrase in text for text in reference_texts):
            match_found.append(True)
        else:
            match_found.append(False)
    
    df['Match Found'] = match_found
    df = df[df['Match Found'] == True]

    match_count = df.groupby('Context Length').size().reset_index(name='Count')
    if(match_count.empty):
        print("No overlap found between the synthetic and reference files.")
    else:
        #match_count['Memorized Span Ratio'] = match_count['Count']/len(reference_texts)
        #print("Length of reference texts: ", len(reference_texts))
        print("Memorized Span Overlap Count:")
        print(tabulate(match_count, headers='keys', tablefmt='psql', showindex=False))

# Alternate version
def compute_phrase_text_overlap_deprecated(synth_file_path, ref_file_path, remove_duplicates=True):

    """
    This function compares the phrases extracted from a synthetic file with those from a reference file,
    calculates the overlap in terms of context length, and prints the overlap ratio.

    Args:
        synth_file_path (str): Path to the synthetic file containing phrases.
        ref_file_path (str): Path to the reference file containing phrases.
        remove_duplicates (bool): Flag to indicate whether to remove duplicates from the dataframes before comparison.
    
    Returns:
        pd.DataFrame: A DataFrame containing the overlap count and ratio for each context length.
    """
    df = pd.read_csv(synth_file_path)    
    ref_df = pd.read_csv(ref_file_path)
    
    for i in [df, ref_df]:
        i['Context Length'] = i['Context Length'].astype(int)
        if(remove_duplicates):
            # Remove duplicates from each dataframe
            i.drop_duplicates(subset=['Entity', 'Phrase', 'Context Length'], inplace=True)

    overlap_df = pd.merge(ref_df, df, on=['Entity', 'Phrase', 'Context Length'], how='inner')
    overlap_count = overlap_df.groupby('Context Length').size().reset_index(name='Overlap Count')
    total_count = ref_df.groupby('Context Length').size().reset_index(name='Total Count')

    # Merge the overlap count with the total count to calculate the ratio
    overlap_ratio = pd.merge(overlap_count, total_count, on='Context Length')
    if(overlap_ratio.empty):
        print("No overlap found between the synthetic and reference files.")
    else:
        overlap_ratio['Overlap Ratio (Synth Freq/Train Freq)'] = overlap_ratio['Overlap Count'] / overlap_ratio['Total Count']

        print("Memorized Span Overlap Ratio:")
        print(tabulate(overlap_ratio, headers='keys', tablefmt='psql', showindex=False))

        return overlap_ratio