from datasets import load_dataset, DatasetDict, concatenate_datasets
import sys
import numpy as np

ds = load_dataset("mattmdjaga/text-anonymization-benchmark-train")

def download_text_anonymization_benchmark(path_to_data):

    tab_dict = DatasetDict()
    
    ds = load_dataset("mattmdjaga/text-anonymization-benchmark-train")
    tab_dict['train'] = ds['train']
    ds = load_dataset("mattmdjaga/text-anonymization-benchmark-val-test")
    tab_dict['validation'], tab_dict['test'] = ds['validation'], ds['test']

    # Defining the control code
    tab_dict = tab_dict.map(lambda x: {'control': "Countries: " + x['meta']['countries'] +", Year: " + str(x['meta']['year'])})
    tab_dict = tab_dict.map(lambda x: {'country': x['meta']['countries']})
    tab_dict = tab_dict.map(lambda x: {'year': x['meta']['year']})

    tab_dict = tab_dict.remove_columns(['annotations'])
    tab_dict['test'] = concatenate_datasets([tab_dict['train'].select(np.random.randint(0, len(tab_dict['train']), size=1000)), tab_dict['test']])
    
    tab_dict.save_to_disk(path_to_data)


download_text_anonymization_benchmark(sys.argv[1])