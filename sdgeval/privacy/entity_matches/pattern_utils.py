import re
import pandas as pd
from tqdm import tqdm
from collections import Counter

def calculate_overlap(f1, f2):
    df1, df2 = pd.read_csv(f1), pd.read_csv(f2)
    list1, list2 = df1['Phrase'].tolist(), df2['Phrase'].tolist()
    set1 = set(list1)
    set2 = set(list2)

    overlap = set1.intersection(set2)
    union = set1.union(set2)
    
    return len(overlap)/len(union), len(overlap), len(union), len(list2)
    
def single_phrase_search_within_document(documents, k = 3):
    context_pattern = re.compile(r'((?:\S+\s+){0,' + str(k) + r'})\[\*\*[^\[]*?\*\*\]((?:\s+\S+){0,' + str(k) + r'})')
    
    all_phrases = []
    for doc in documents:
        matches = context_pattern.finditer(doc)
        for match in matches:
            
            pattern = match.group(0).strip()
            if('Date' in pattern):
                continue
            combined_match = f"{pattern}".strip()
            all_phrases.append(combined_match)
    
    return all_phrases

def search_phrase(path_to_file, text_field, k_adj_values = [1, 2, 3, 4], save_file_path = None):
    try:
        df = pd.read_csv(path_to_file)
    except:
        df = path_to_file
    phrases, k_adj_count = [], []
    for k_adj in k_adj_values:
        new_phrase = single_phrase_search_within_document(df[text_field].tolist(), k = k_adj)
        phrases = phrases + new_phrase
        k_adj_count = k_adj_count + [k_adj]*len(new_phrase)
    
    if(not save_file_path):
        save_file_path = 'phrases_' + path_to_file[path_to_file.rfind('/') + 1:] + '.csv'
    df = pd.DataFrame({'Phrase': phrases, 'Context Length': k_adj_count})
    df.to_csv(save_file_path)
    print(df.head())
    return phrases

def extract_tokens(documents, alpha_only = False):
    pattern = re.compile(r'\[\*\*[^\[]*?\*\*\]')
    all_tokens = []
    
    for doc in documents:
        tokens = pattern.findall(doc)
        all_tokens.extend(tokens)
    if(alpha_only):
        all_tokens = [tok for tok in all_tokens if(bool(re.search(r'[A-Za-z]', tok)))]
    return all_tokens

def return_freq_dict(tokens, top_n = 50, is_filter = False):
    frequency_dict = Counter(tokens)
    if(is_filter):
        frequency_dict = {key: value for key, value in frequency_dict.items() if value <= 200}
    if(top_n):
        sorted_dict = dict(sorted(frequency_dict.items(), key=lambda item: item[1], reverse=True)[:top_n])
    else:
        sorted_dict = dict(sorted(frequency_dict.items(), key=lambda item: item[1], reverse=True))
    return sorted_dict

def find_phi(df, text_field):
    documents = df[text_field].tolist()
    tokens = extract_tokens(documents)
    tokens = [tok for tok in tokens if(bool(re.search(r'[A-Za-z]', tok)))]
    frequency_dict = return_freq_dict(tokens, top_n = None)
    return frequency_dict

    