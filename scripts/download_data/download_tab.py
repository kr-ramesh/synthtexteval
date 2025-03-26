from datasets import load_dataset, DatasetDict, concatenate_datasets
import sys
import numpy as np

ds = load_dataset("mattmdjaga/text-anonymization-benchmark-train")

np.random.seed(42)

def download_text_anonymization_benchmark(path_to_data, path_to_coref_dataset):

    tab_dict = DatasetDict()
    
    ds = load_dataset("mattmdjaga/text-anonymization-benchmark-train")
    tab_dict['train'] = ds['train']
    ds = load_dataset("mattmdjaga/text-anonymization-benchmark-val-test")
    tab_dict['validation'], tab_dict['test'] = ds['validation'], ds['test']

    # Defining the control code
    tab_dict = tab_dict.map(lambda x: {'control': "Countries: " + x['meta']['countries'] +", Year: " + str(x['meta']['year'])})
    tab_dict = tab_dict.map(lambda x: {'country': x['meta']['countries']})
    tab_dict = tab_dict.map(lambda x: {'year': x['meta']['year']})

    # Creates data for benchmarking for the coref task
    if(path_to_coref_dataset):
        df = tab_dict['train'].to_pandas()
        df.to_csv(path_to_coref_dataset + '/real_tab_train_coref.csv')
        df = concatenate_datasets([tab_dict['test'], tab_dict['validation']]).to_pandas()
        df.to_csv(path_to_coref_dataset+ '/real_tab_test_coref.csv')
    train_concat = tab_dict.remove_columns(['annotations'])['train']
    # Sampling from the training data as the test set is too small for TAB
    tab_dict['test'] = concatenate_datasets([train_concat.select(np.random.randint(0, len(train_concat), size=1000)), tab_dict['test']])
    tab_dict.save_to_disk(path_to_data)

download_text_anonymization_benchmark(sys.argv[1], sys.argv[2])