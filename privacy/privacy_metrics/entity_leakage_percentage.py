def entities_in_paragraph(paragraph: str, entities: list) -> dict:
    """
    Checks whether each entity in `entities` appears in `paragraph`.
    
    :param paragraph: The text to be searched.
    :param entities: A list of entity strings to look for in the paragraph.
    :return: A dictionary with each entity as a key and a boolean value indicating presence.
    """
    results = {}
    for entity in entities:

        results[entity] = entity in paragraph
    return results

if __name__ == "__main__":
    paragraph_text = ""
   
    
    entities_to_check = ""
    
    presence_dict = entities_in_paragraph(paragraph_text, entities_to_check)
    
    # Print out the result for each entity
    for entity, found in presence_dict.items():
        print(f"Entity '{entity}' found? {found}")
