def entities_in_paragraph(paragraph: str, entities: list) -> dict:
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
