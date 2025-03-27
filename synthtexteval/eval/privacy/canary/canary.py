import os, sys, pickle, argparse, random, math, re, pandas as pd
import torch
import peft
from synthtexteval.eval.privacy.canary.generate_candidates import read_candidate_data
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from synthtexteval.generation.controllable.load_models import load_model_tokenizer
from dataclasses import dataclass, field
from tqdm import tqdm

@dataclass
class CanaryArguments:
    """
    Arguments for initializing and running the canary attack evaluation.
    """
    insertion_N: str = field(default="1000", metadata={"help": "The number of insertions for each canary."})
    path_to_dataset: str = field(default=None, metadata={"help": "The path to data to be modified."})
    text_column: str = field(default=None, metadata={"help": "The column in the csv column containing the text to be modified.."})
    model_name: str = field(default="", metadata={"help": "The path to the  original model on the HF Hub."})
    path_to_model: str = field(default="", metadata={"help": "The path to model to be evaluated for a canary attack."})
    path_to_save_dataset: str = field(default="", metadata={"help": "The path where the modified data should be saved."})
    canary_file_path: str = field(default="canaries.txt", metadata={"help": "Path to the text file containing the canary insertion data."})
    candidates_file_path: str = field(default="candidates.csv", metadata={"help": "Path to the text file containing the canary candidates."})
    create_dataset: bool = field(default=False, metadata={"help": "If True, creates a canary dataset. If False, performs a canary attack evaluation."})
    lora_enabled: bool = field(default=False, metadata={"help": "Whether LoRA training is used."})

class Canary(Dataset):
    """
    Class for creating and evaluating canary attacks.
    """
    def __init__(self, args):
        canary_dict = read_candidate_data(args.canary_file_path)
        self.canary_list = canary_dict['candidate_sentence']
        if(args.create_dataset):
            # For creating the dataset for training the model on the canaries.
            self.data = pd.read_csv(args.path_to_dataset)
            self.text_column = args.text_column
            self.insertion_N = args.insertion_N
            self.all_indices_canary = []
            self.total_insertions = int(len(self.canary_list))*int(self.insertion_N)
            self.create_canary_list()
            self.path_to_save_dataset = args.path_to_save_dataset
        else:
            # For evaluating the model for the leakage of canaries
            self.candidates_file_path = args.candidates_file_path
            self.path_to_model = args.path_to_model
            self.model, self.tokenizer = load_model_tokenizer(model_name = args.model_name)
            if(args.lora_enabled):
                print("Lora enabled...")
                self.model.load_adapter(self.path_to_model)
                print("Model loaded...")

    def create_canary_list(self): 
        limit = int(self.total_insertions/len(self.canary_list))
        self.canary_list = [self.canary_list[i] for i in range(len(self.canary_list)) for x in range(limit)]
        random.shuffle(self.canary_list)

    def insert_canary_in_middle(self, item, canary):
        
        periods_indices = [m.start() for m in re.finditer(r'\.', item)]
        # Select a random index from the list of periods indices
        try:
            random_index = random.choice(periods_indices)
        except:
            print("Period not found.")
            return item + '. ' + canary
        
        modified_text = item[:random_index + 1] + " " + canary + item[random_index + 1:]

        return modified_text

    def create_dataset_training(self):

          """
          This function creates a dataset for training the model on the canaries. 
          It inserts the canaries insertion_N times in the text_column of the dataset. 
          """

          data_text = self.data[self.text_column].tolist()
          canaries = ['no_canary' for i in range(len(data_text))]
          
          print("Processing canary insertions...")
          #Defines all the indices of the dataset at which canaries will be inserted.
          self.all_indices_canary = random.sample(range(0, len(data_text)-1), self.total_insertions)
          for data_index, canary_index in zip(self.all_indices_canary, range(0, len(self.canary_list))):
            #Modifies the data point at which the canary is inserted.
            data_text[data_index] = self.insert_canary_in_middle(data_text[data_index], canary = self.canary_list[canary_index])
            #Modifies the column to show which canary appears in that index
            canaries[data_index] = self.canary_list[canary_index]
          
          print("Done with modifications.")
          
          self.data[self.text_column] = data_text
          self.data['Canary'] = canaries
          try:
            os.mkdir(self.path_to_save_dataset[:self.path_to_save_dataset.rfind('/')])
          except:
            print("Directory exists")
          self.data.to_csv(self.path_to_save_dataset)


    def create_dataset_testing(self, canary, candidate_list):

          """
          This function creates a dataset for evaluating the model on the canaries.
          It evaluates the perplexity of the model on the canaries and ranks them based on the perplexity.
          Higher perplexity typically indicates lower chances of leakage of the canary.
          """

          #TODO: Consider sampling for words similar to the entity at a given index of the canary using word embedding based methods
          #TODO: (continued) Create duplicate dataset using this list of words

          #Replacing entity_index with words from a list.
          candidate_list = candidate_list + [canary]
          encoded_texts = [self.tokenizer.encode(text) for text in candidate_list]

          return list(zip(candidate_list, encoded_texts))


    def calculate_exposure(self, TOTAL_CANDIDATES, canary, candidate_list):

          """Calculates the exposure of the model for a given canary.
          The exposure is calculated as the log of the rank of the canary in the sorted list of perplexities."""

          self.data = self.create_dataset_testing(canary, candidate_list)

          #print("Dataset is ", self.data) #Debugging
          dataloader = DataLoader(dataset=self.data,
                                  shuffle=False,
                                  batch_size=1,
                                  collate_fn=self.collate)


          #Use the test_model_perplexity function to calculate the rank and perplexity for the entire canary dataset
          unsorted_ppls = {}

          for batch in tqdm(dataloader):

                #print(batch)

                batch_text = list(map(lambda x: x[0], batch))
                batch_encoded_text = list(map(lambda x: x[1], batch))
                batch_ppl = self.test_model_perplexity(batch_encoded_text)

                unsorted_ppls.update(dict(zip(batch_text, batch_ppl)))

          sorted_ppls = {k: (i+1, v) for i, (k, v) in enumerate(sorted(unsorted_ppls.items(), key=lambda item: item[1]))}

          #Calculate the exposure
          #print(sorted_ppls)
          canary_rank, canary_ppl = sorted_ppls[canary]
          canary_exposure = math.log(TOTAL_CANDIDATES, 2) - math.log(canary_rank, 2)

          return canary_exposure, canary_rank, canary_ppl, sorted_ppls

    def test_model_perplexity(self, batch_sentence):

          """Calculates and returns the sorted rank
          and perplexity of the test set for the canary attack."""

          PAD_TOKEN_ID = self.tokenizer.pad_token
          PAD_TOKEN_ID = self.tokenizer.encode(PAD_TOKEN_ID)[0]

          criterion = torch.nn.CrossEntropyLoss(ignore_index = PAD_TOKEN_ID, reduction='none')

          #Calculate the perplexity based on the loss

          #Sort the list of perplexities
          batch_size = len(batch_sentence)

          with torch.no_grad():  # no tracking history

              source = list(map(lambda x: torch.tensor(x[:-1]).type(torch.int64), batch_sentence))
              target = list(map(lambda x: torch.tensor(x[1:]).type(torch.int64), batch_sentence))
              seq_lens = list(map(lambda x: len(x) - 1, batch_sentence))
              source = pad_sequence(source, batch_first=True, padding_value=PAD_TOKEN_ID)#.to(0)
              target = pad_sequence(target, batch_first=True, padding_value=PAD_TOKEN_ID)#.to(Z)

              attention_mask = (source != PAD_TOKEN_ID).type(torch.int64)#.to(device)

              outputs = self.model(input_ids=source, attention_mask=attention_mask)
              logits = outputs.logits.reshape((outputs.logits.shape[0]*outputs.logits.shape[1], -1))
              target = target.view(-1)
              total_loss = criterion(logits, target).reshape((batch_size, -1)).cpu().numpy()

              ppls = []
              for loss in total_loss:
                  sum_loss = sum(loss)
                  ntokens = sum([l!=0 for l in loss])
                  ppls.append(math.exp(sum_loss/ntokens))

          return ppls

    def collate(self, unpacked_data):
          return unpacked_data
    
    def canary_attack_eval(self, canary_results_output_path):

        """
        This function evaluates the model for a canary attack.
        It calculates the exposure of the model for each canary and ranks them based on the exposure.
        """
        
        df = pd.read_csv(self.candidates_file_path)
        ent_dict = df.groupby('Subject', sort=False)['Candidate'].apply(list).to_dict()
        ent_dict = {k:v for k, v in zip(self.canary_list, ent_dict.values())}
        sorted_ppls, sorted_models = [], []
        for canary in ent_dict:
            c_exp, c_rank, c_ppl , sorted_ppl = self.calculate_exposure(TOTAL_CANDIDATES = len(ent_dict[canary]), canary = canary, candidate_list = ent_dict[canary])
            print('Exposure:', c_exp)
            print('Rank:', c_rank)
            print('Perplexity', c_ppl)
            print('\n\n')
            sorted_ppls.append({k : [sorted_ppl[k][1]] for k in sorted_ppl.keys()})
            sorted_models.append(self.path_to_model)
        df = pd.DataFrame({'Sorted PPL': sorted_ppls, 'Models':sorted_models})
        df.set_index('Models', inplace=True)
        df.to_csv(canary_results_output_path, mode = 'a')


"""if __name__ == "__main__":
        args = CanaryArguments()
        if(args.create_dataset):
            obj = Canary(args)
            obj.create_dataset_training()
        else:
            print("Privacy evaluation")
            obj = Canary(args)
            obj.canary_attack_eval('results.csv')"""