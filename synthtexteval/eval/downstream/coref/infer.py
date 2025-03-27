import json
import os
import logging
import random
from collections import OrderedDict, defaultdict
import numpy as np
import torch
from synthtexteval.eval.downstream.coref.coref_bucket_batch_sampler import BucketBatchSampler
from synthtexteval.eval.downstream.coref.data import get_dataset
from synthtexteval.eval.downstream.coref.metrics import CorefEvaluator, MentionEvaluator
from synthtexteval.eval.downstream.coref.utils import extract_clusters, extract_mentions_to_predicted_clusters_from_clusters, extract_clusters_for_decode
from synthtexteval.eval.downstream.coref.conll import evaluate_conll
import json
from tqdm import tqdm


# logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(self, args, tokenizer):
        self.args = args
        self.eval_output_dir = args.output_dir
        self.tokenizer = tokenizer

    def inference(self, model, tb_writer=None, global_step=None):
        if self.args.predict_file_write == None:
            return
        eval_dataset = get_dataset(self.args, tokenizer=self.tokenizer, file_path=self.args.predict_file)

        if self.eval_output_dir and not os.path.exists(self.eval_output_dir) and self.args.local_rank in [-1, 0]:
            os.makedirs(self.eval_output_dir)

        # Note that DistributedSampler samples randomly
        # eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = BucketBatchSampler(eval_dataset, max_total_seq_len=self.args.max_total_seq_len, batch_size_1=True)

        # Eval!
        # logger.info("***** Running evaluation *****")
        # logger.info("  Examples number: %d", len(eval_dataset))
        model.eval()

        post_pruning_mention_evaluator = MentionEvaluator()
        mention_evaluator = MentionEvaluator()
        coref_evaluator = CorefEvaluator()
        losses = defaultdict(list)
        doc_to_prediction = {}
        doc_to_subtoken_map = {}

        for (doc_key, subtoken_maps), batch in tqdm(eval_dataloader):

            batch = tuple(tensor.to(self.args.device) for tensor in batch)
            input_ids, attention_mask, gold_clusters = batch # gold_clusters would likely be empty
            # gold_clusters = []
            with torch.no_grad():
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                gold_clusters=gold_clusters,
                                return_all_outputs=True)
                loss_dict = outputs[-1]

            if self.args.n_gpu > 1:
                loss_dict = {key: val.mean() for key, val in loss_dict.items()}

            for key, val in loss_dict.items():
                losses[key].append(val.item())

            outputs = outputs[1:-1]

            batch_np = tuple(tensor.cpu().numpy() for tensor in batch)
            outputs_np = tuple(tensor.cpu().numpy() for tensor in outputs)
            for output in zip(*(batch_np + outputs_np)):
                gold_clusters = output[2]
                gold_clusters = extract_clusters(gold_clusters)
                mention_to_gold_clusters = extract_mentions_to_predicted_clusters_from_clusters(gold_clusters)
                gold_mentions = list(mention_to_gold_clusters.keys())

                starts, end_offsets, coref_logits, mention_logits = output[-4:]

                max_antecedents = np.argmax(coref_logits, axis=1).tolist()
                mention_to_antecedent = {((int(start), int(end)), (int(starts[max_antecedent]), int(end_offsets[max_antecedent]))) for start, end, max_antecedent in
                                         zip(starts, end_offsets, max_antecedents) if max_antecedent < len(starts)}

                predicted_clusters, _ = extract_clusters_for_decode(mention_to_antecedent)
      
                candidate_mentions = list(zip(starts, end_offsets))

                mention_to_predicted_clusters = extract_mentions_to_predicted_clusters_from_clusters(predicted_clusters)
                predicted_mentions = list(mention_to_predicted_clusters.keys())
                post_pruning_mention_evaluator.update(candidate_mentions, gold_mentions)
                mention_evaluator.update(predicted_mentions, gold_mentions)
                coref_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted_clusters,
                                       mention_to_gold_clusters)
                doc_to_prediction[doc_key] = predicted_clusters
                doc_to_subtoken_map[doc_key] = subtoken_maps


        read_fname = self.args.predict_file # 
        dataset_id = os.path.basename(read_fname).replace('synth_', '').replace('.jsonlines', '')
        
        write_fname = self.args.predict_file_write 
        # reverse the clusters
        doc_to_prediction_word = eval_dataset._reverse_tokenize_clusters(eval_dataset.pretokenized_examples, doc_to_prediction)

        to_write = []
        with open(read_fname, 'r') as f:
            for line in f:
                d = json.loads(line.strip())
                doc_key = d["doc_key"]
                if doc_key not in doc_to_prediction_word:
                    d['clusters'] = []
                else:
                    d['clusters'] = doc_to_prediction_word[doc_key]
               
                to_write.append(d)
        with open(write_fname, 'w') as f:
            for line in to_write:
                f.write(json.dumps(line))
                f.write("\n")

        