import nltk, re, pandas as pd, numpy as np, operator, os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict, Counter

nltk.download('stopwords')
nltk.download('wordnet')

STOPWORDS = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
blacklist_words = ['none', 'none.', 'nonexistent', 'nonexistant']

#TODO: Clean up and create a filter class.

def preprocess_text(text):
    """
    Preprocesses the text data by removing special characters, converting to lowercase, removing stopwords, and lemmatizing.
    """
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    text = " ".join([word for word in str(text).split() if word not in STOPWORDS])
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])

    return text

def check_repeats(text, predefined_percentage = 0.15, freq_count = 20):
    """
    Checks if any word is repeated more times than the frequency_limit threshold.
    In addition to this, if any word exceeds the predefined_percentage of words in the document, we do not keep that data instance.
    """
    words = text.split()
    total_words = len(words)
    word_counts = Counter(words)
    for count in word_counts.values():
        if count > predefined_percentage * total_words or count > freq_count:
            return False
    return True

def discard_data(df, discard_limit = 20, frequency_limit = 10, text_field = "CMNT_TXT"):
    
    """
    Discards data based on the following criteria:
    1. If the text field has 'None' repeated, it is removed.
    2. If the text field has more than 400 characters, it is removed.
    3. If any given word is repeated more times than the frequency_limit threshold, do not keep that data instance.
    4. If any word exceeds the predefined_percentage of words in the document, we do not keep that data instance.
    """

    df[text_field] = df[text_field].apply(lambda x: x.replace('None.', ''))
    df[text_field] = df[text_field].apply(lambda x: x[:x.index('None None')] if ('None None') in x else x)
    df[text_field] = df[text_field].apply(lambda x: x.replace('None.', ''))

    #Removes all extra whitespaces and newlines
    texts = [re.sub(r'\s+', ' ', t.strip()) for t in df[text_field].tolist()]
    texts = [preprocess_text(t) for t in texts]

    texts_index = [True if len(set(x.lower().split()))>discard_limit else False for x in texts]
    texts_index = [texts_index[texts.index(x)] if len(x)>=400 else False for x in texts]
    df = df[texts_index]
    
    #If any given word is repeated more times than the frequency_limit threshold, do not keep that data instance
    #In addition to this, if any word exceeds the predefined_percentage of words in the document, we do not keep that data instance
    df= df[df[text_field].apply(check_repeats)]    
        
    return df

def process_df(df, text_column = 'output_text'):

    df[text_column] = df[text_column].apply(lambda x: x[:x.rfind('.')+1])
    
    return df