import random
import pandas as pd
from faker import Faker
from faker.providers import BaseProvider
import string

fake_email_len = random.randrange(8, 15)

def generate_random_string(length):
    """
    Generate a random string of fixed length. 
    """
    # Use string.ascii_letters to include both lowercase and uppercase letters
    letters = string.ascii_letters
    random_string = ''.join(random.choice(letters) for _ in range(length))
    return random_string.lower()

class CustomProvider(BaseProvider):
    """
    Custom provider for generating fake data.
    """
    def case_number(self):
        return f"{random.randint(2000, 2023)}-{random.randint(1000, 9999)}"
    def email(self):
        return f"{generate_random_string(fake_email_len)}{random.randint(0, 999)}@email.com"

def return_fake_item(candidate_type, fake):
    """
    Return a fake item based on the candidate type.
    """
    candidate_type = candidate_type.lower()
    if(candidate_type == 'Name'.lower()):
        return fake.unique.name()
    if(candidate_type == 'Case Number'.lower()):
        return fake.unique.case_number()
    if(candidate_type == 'Address'.lower()):
        return fake.unique.street_address()
    if(candidate_type == 'Date'.lower()):
        return fake.date_between()
    if(candidate_type == 'Email'.lower()):
        return fake.unique.email()
    if(candidate_type == "Number".lower()):
        return fake.phone_number()    

def read_candidate_data(file_path):
    """
    Read candidate data from a file.
    """
    extracted_data = {'candidate_sentence': [], 'candidate':[], 'candidate_type': []}
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(' || ')
            if len(parts) == 3:
                candidate_sentence, candidate, candidate_type = parts
                extracted_data['candidate_sentence'].append(candidate_sentence)
                extracted_data['candidate'].append(candidate)
                extracted_data['candidate_type'].append(candidate_type)
                
    return extracted_data

def generate_candidates(candidate_output_file_path, candidate_dict, n_candidates = 100):
    """
    Generates candidate data similar to the canary in terms of format.
    """
    fake = Faker()
    fake.add_provider(CustomProvider)
    
    candidates, template_sentences, candidate_types = candidate_dict['candidate'], candidate_dict['candidate_sentence'], candidate_dict['candidate_type']
    replacement_candidates = {k:[] for k in candidates}
    
    for ind in range(len(candidates)):
        for i in range(n_candidates):
        #try:
            replacement_candidates[candidates[ind]].append(return_fake_item(candidate_types[ind], fake))
        #except Exception as e:
        #    print(e)
        #    i = i-1

    candidate_sentences = []

    for ind, candidate in enumerate(candidates):
        for replace_sent in replacement_candidates[candidate]:
            candidate_sentences.append((candidate, template_sentences[ind].replace(candidate, str(replace_sent))))

    df = pd.DataFrame(candidate_sentences, columns=["Subject", "Candidate"])
    df.to_csv(candidate_output_file_path, index=False)

#Usage example
"""
    fake = Faker()
    fake.add_provider(CustomProvider)
    candidate_dict = read_candidate_data(file_path = 'candidates.txt')
    generate_candidates('candidates.csv', candidate_dict)
"""