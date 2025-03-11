from datasets import load_dataset, DatasetDict

ds = load_dataset("mattmdjaga/text-anonymization-benchmark-train")

def download_text_anonymization_benchmark():

    tab_dict = DatasetDict()
    
    ds = load_dataset("mattmdjaga/text-anonymization-benchmark-train")
    tab_dict['train'] = ds['train']
    ds = load_dataset("mattmdjaga/text-anonymization-benchmark-val-test")
    tab_dict['validation'], tab_dict['test'] = ds['validation'], ds['test']

    # Defining the control code
    tab_dict = tab_dict.map(lambda x: {'control': "Countries: " + x['meta']['countries'] +", Year: " + str(x['meta']['year'])})
    tab_dict = tab_dict.map(lambda x: {'country': x['meta']['countries']})
    tab_dict = tab_dict.map(lambda x: {'year': x['meta']['year']})

    # Save the dataset to disk
    tab_dict.save_to_disk("./generate/data/tab")
    
download_text_anonymization_benchmark()