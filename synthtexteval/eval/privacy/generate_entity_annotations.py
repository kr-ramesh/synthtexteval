# This code provides functionality for generating silver entity annotations using pre-trained NER models from HuggingFace and Spacy.
# The generated annotations are saved in a pickle file and can be used for further use in the pipeline.
import spacy
import pickle
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

nlp = spacy.load('en_core_web_sm')

def get_entities_hf(text, model_name = "dslim/bert-base-NER"):

    """
    Get entities from the text using the HuggingFace NER model.
    Args:
        text (str): The text to extract entities from.
        model_name (str): The name of the HuggingFace NER model to use.
    Returns:
        entities (lst): List of entities.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    ner_results = nlp(text)
    entities = [(item['entity'], item['score'], item['word']) for ner_text in ner_results for item in ner_text]

    return entities
    
def get_entities_spacy(text):

    """
    Get entities from the text using the Spacy NER model.
    """
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    return entities

def generate_entity_annotations(entity_annotations_path, df, text_column = 'text', entity_extraction = 'spacy'):
    """
    Generate entity annotations for the given dataframe using an existing NER model.
    Args:
        df (pd.DataFrame): The dataframe containing the data.
    Returns:
        entities (lst): List of entities.
    """
    texts = df[text_column].tolist()
    

    entities = []
    if(entity_extraction == 'spacy'):
        for text in texts:
            entities = entities + get_entities_spacy(text)

    elif(entity_extraction == 'hf'):
        entities = get_entities_hf(texts)
    
    # Save to a pickle file
    with open(entity_annotations_path, 'wb') as f:
        pickle.dump(entities, f)
    #return entities

# Example usage
#df = pd.DataFrame({'text': ['Apple is looking at buying U.K. startup for $1 billion', 'San Francisco considers banning sidewalk delivery robots']})
#generate_entity_annotations('entities-silver.pkl', df, text_column = 'text', entity_extraction = 'spacy')