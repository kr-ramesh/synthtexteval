import argparse
import os
import pandas as pd
import pickle
from pattern_utils import search_phrase, calculate_overlap, find_phi

#TODO: Change this so that it accepts a list of entities from the user (assumption is that user has already extracted the PHI.)
"""Usage:
    
    python pattern_main.py --ref_file_path "/export/fs06/kramesh3/psd/princeton_mimic_10ICD_DP_inf.csv" /
    --phrase_save_file_path "princeton_mimic_10ICD_DP_inf.csv" /
    --phrase_save_file_dir "phrases-updated" /
    --pattern_search
    
    python pattern_main.py --train_file_path "princeton_mimic_10ICD_train.csv" /
    --ref_file_path "princeton_mimic_10ICD_DP_inf.csv" /
    --overlap_save_file_path "overlap-results/overlap-results-final-updated.csv" /
    --pattern_overlap

    python pattern_main.py --ref_file_path "/export/fs06/kramesh3/psd/princeton_mimic_10ICD_DP_inf.csv" /
    --phi_save_file_dir "phi-updated" /
    --find_phi
"""

def parse_arguments():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process command-line arguments.")
    
    # Add arguments
    parser.add_argument('--train_file_path', type=str, required=False, help='Path to the training data.')
    parser.add_argument('--ref_file_path', type=str, required=False, help='Path to the reference data to compare against.')
    
    parser.add_argument('--phrase_save_file_dir', type=str, required=False, default='phrases-updated', help='Directory where phrase searches are saved.')
    parser.add_argument('--phrase_save_file_path', type=str, required=False, help='Path to save the results for a phrase search.')
    
    parser.add_argument('--overlap_save_file_path', type=str, required=False, help='Path to save the results for a phrase overlap search.')
    
    parser.add_argument('--phi_save_file_dir', type=str, required=False, default='phi-updated', help='Directory where the PHI results are saved.')
    
    parser.add_argument('--pattern_search',  action="store_true", help='Pattern search option.')
    parser.add_argument('--pattern_overlap',  action="store_true", help='Pattern overlap with training data.')
    parser.add_argument('--find_phi',  action="store_true", help='Find PHI from the reference data.')

    # Parse the arguments
    args = parser.parse_args()
    return args

def main(args):
    if(os.path.exists(args.phi_save_file_dir) == False):
            os.mkdir(args.phi_save_file_dir)
    if(os.path.exists(args.phrase_save_file_dir) == False):
            os.mkdir(args.phrase_save_file_dir)

    if(args.pattern_search):
        #Searches for exact matches of contexts PHI appears in for both the training and reference data
        #TODO: Change this to: extract context lengths for all PHI for a given data file and save it to directory. Calculate overlap.
        print("Pattern search")
        df = pd.read_csv(args.ref_file_path)
        phrases_found = search_phrase(path_to_file = df,
                                      text_field = 'text',
                                      save_file_path = args.phrase_save_file_dir + '/' + args.phrase_save_file_path)
        print(len(phrases_found))
    
    if(args.pattern_overlap):
        #Using the results from the pattern search to estimate privacy leakage
        print("Pattern overlap")
        overlap_ratio, overlap_count, union_count, input_doc_len = calculate_overlap(args.phrase_save_file_dir + '/' + args.train_file_path, args.phrase_save_file_dir + '/' + args.ref_file_path)
        print(overlap_ratio, overlap_count, union_count)
        result = {'Train File': args.train_file_path, 'Ref File': args.ref_file_path, 'Ratio':overlap_ratio, 'Unique Overlap Count':overlap_count, 'Unique Union Count': union_count, 'Synth Doc Len':input_doc_len}
        result = pd.DataFrame([result])
        if(os.path.exists(args.overlap_save_file_path)):
            result.to_csv(args.overlap_save_file_path, mode = 'a', header = None, index = None)
        else:
            result.to_csv(args.overlap_save_file_path, index = None)
    
    if(args.find_phi):
        #Searches for PHI from a given file and saves it with corresponding entity type
        print("Finding PHI from: ", args.ref_file_path)
        df = pd.read_csv(args.ref_file_path)
        freq_dict = find_phi(df, text_field = 'text')
        pickle_filename = args.phi_save_file_dir + '/' + args.ref_file_path[args.ref_file_path.rfind("/") + 1:-4] + '.pkl'
        with open(pickle_filename, 'wb') as file:
            pickle.dump(freq_dict, file)
        freq_dict = list(set(freq_dict))
        print("Length of PHI set: ", len(freq_dict))
        

if __name__ == '__main__':
    parsed_args = parse_arguments()
    main(parsed_args)