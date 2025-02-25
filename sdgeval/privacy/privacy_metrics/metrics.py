import re

def entity_leakage(paragraph: str, entities: list) -> dict:
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
