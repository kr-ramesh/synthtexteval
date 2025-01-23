def entity_presence(paragraph: str, entities: list, case_sensitive: bool = False) -> dict:
    results = {}
    if not case_sensitive:
        paragraph_lower = paragraph.lower()
        for entity in entities:
            entity_lower = entity.lower()
            results[entity] = entity_lower in paragraph_lower
    else:
        for entity in entities:
            results[entity] = entity in paragraph
    
    return results


def entity_coverage_percentage(paragraph: str, entities: list, case_sensitive: bool = False) -> float:
    
    if not paragraph:
        return 0.0  
    search_paragraph = paragraph if case_sensitive else paragraph.lower()

    intervals = []
    for entity in entities:
        search_entity = entity if case_sensitive else entity.lower()
        start_index = 0
        
        while True:

            found_at = search_paragraph.find(search_entity, start_index)
            if found_at == -1:
                break
            end_at = found_at + len(search_entity)
            intervals.append((found_at, end_at))
            
   
            start_index = end_at
    

    if not intervals:
        return 0.0

    intervals.sort(key=lambda x: x[0])
    merged_intervals = []
    
    current_start, current_end = intervals[0]
    
    for i in range(1, len(intervals)):
        interval_start, interval_end = intervals[i]
        
        if interval_start <= current_end:
            current_end = max(current_end, interval_end)
        else:
            merged_intervals.append((current_start, current_end))
            current_start, current_end = interval_start, interval_end
    

    merged_intervals.append((current_start, current_end))
    covered_length = sum((end - start) for start, end in merged_intervals)
    coverage_percentage = (covered_length / len(paragraph)) * 100
    return coverage_percentage


if __name__ == "__main__":
    paragraph_text = ""
    
    entities_to_check = ""
    
    presence_dict = entity_presence(paragraph_text, entities_to_check, case_sensitive=False)
    coverage_percent = entity_coverage_percentage(paragraph_text, entities_to_check, case_sensitive=False)
    
    print("Entity Presence:")
    for entity, found in presence_dict.items():
        print(f"  - '{entity}' found? {found}")
    
    print(f"\nCoverage Percentage: {coverage_percent:.2f}%")
