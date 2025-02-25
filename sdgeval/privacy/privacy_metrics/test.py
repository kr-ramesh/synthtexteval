from sdgeval.privacy.privacy_metrics.metrics import entity_leakage
from sdgeval.privacy.privacy_metrics.metrics import search_phrase
import pandas as pd

fake_text = """Investigative journalist Rachel Marin thought she had seen it all, but when a whistleblower from Velkor Industries 
leaked classified documents, she knew she was onto something big. The files detailed secret offshore accounts linked to Senator 
Alan Prescott, a rising political star with close ties to Northbridge Capital, a major investment firm based in Chicago, Illinois.  

As Rachel dug deeper, she uncovered a web of corruption involving the private security contractor Blackthorne Solutions, which had 
recently secured a multi-billion dollar government contract. Her only ally was former FBI analyst Daniel Kessler, who warned her 
that powerful people wouldn’t let the truth come out easily. With the deadline for publication approaching and threats escalating, 
Rachel had to decide—expose the conspiracy and risk everything, or walk away before it was too late."""

fake_entities = [
    "Rachel Marin",
    "Velkor Industries",
    "Senator Alan Prescott",
    "Northbridge Capital",
    "Chicago",
    "Illinois",
    "Blackthorne Solutions",
    "Daniel Kessler"
]

#print(entity_leakage(fake_text, fake_entities))

t_df = pd.DataFrame({'text': [fake_text]})
text_field = 'text'
search_phrase(df = t_df, patterns = fake_entities, max_window_len = 3, text_field = 'text')
