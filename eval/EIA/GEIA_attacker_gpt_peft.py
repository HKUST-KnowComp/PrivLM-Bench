'''
Attacker implementation from the paper
Sentence Embedding Leaks More Information than You Expect: Generative Embedding Inversion Attack to Recover the Whole Sentence
https://arxiv.org/pdf/2305.03010.pdf
'''

import wandb
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(CURRENT_DIR)
from EIA_config import attacker_config

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM
import argparse
from dataset.data_process import get_sent_list
from decode_utils import eval_on_batch
import json
from decode_eval import eval_eia
from GEIA_attacker import GEIA_base_attacker
from peft import  PeftModel

### generative DP_trainers
class GEIA_attacker_gpt2_peft(GEIA_base_attacker):
    def __init__(self,
                 model,
                 tokenizer,
                 attacker_config,
                 config_str,
                 model_name,
                 dp_str,
                 need_proj = True,
                 dataset_name = 'qnli',
                 tuning_method = 'finetune',
                 num_virtual_tokens = 15,
                ):
        ### Init for victim models
        super().__init__(model, tokenizer, attacker_config, config_str,
                 model_name, dp_str, need_proj = need_proj,
                 dataset_name = dataset_name, tuning_method = tuning_method, num_virtual_tokens = num_virtual_tokens)
        self.adapter_name = get_adapter_name(tuning_method)
        self.save_path = os.path.join(BASE_DIR, 'checkpoints_gpt_all',config_str,
                                    dataset_name + '_' + model_name + '_' + tuning_method + '_' + dp_str + '_attacker')
        self.projection_save_path = os.path.join(BASE_DIR, 'checkpoints_gpt_all',config_str,
                                    dataset_name + '_' + model_name + '_' + tuning_method + '_' + dp_str + '_attacker_projection')


    def get_batch_embedding(self, batch_text):
        ### get embeddings from model
        with torch.no_grad():
            tokenizer = self.tokenizer
            tokenizer.pad_token = tokenizer.eos_token
            if(self.tokenizer_len):
                inputs = tokenizer(batch_text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.tokenizer_len)
            else:
                inputs = tokenizer(batch_text, return_tensors='pt', padding=True, truncation=True)

            input_ids = inputs['input_ids']  # tensors of input ids
            input_ids = input_ids.to(self.device)
            output = self.model(input_ids = input_ids,output_hidden_states=True, return_dict=True)
            hidden = output.hidden_states  # tuple of hidden states
            last_hidden_state = hidden[-1]  # tensor of shape (batch_size, seq_len, hidden_size)
            batch_embedding = self.get_embedding_from_hidden(last_hidden_state, mean_pooling=True)  # tensor of shape (batch_size, hidden_size)
        return batch_embedding  ##return (batch_size, hidden_size)
    

def get_adapter_name(tuning_method):
    if tuning_method == 'prefix-tuning':
        adapter_name = 'gpt2_casual_lm_pref'
    elif tuning_method == 'prompt-tuning':
        adapter_name = 'gpt2_casual_lm_perf'
    else:
        raise NotImplementedError
    return adapter_name



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training external NN as baselines')
    parser.add_argument('--porj_path', type=str, default='/home/data/hlibt/p_bench', help='project path')
    args = parser.parse_args()

    project_path = args.porj_path
    #tuning_method = 'prompt-tuning' ### can be 'prefix-tuning' or 'prompt-tuning'
    lr = 1e-2
    batch_size = 4
    n_accumulation_steps = 256
    freeze_embedding = 'True'
    epochs = 5
    target_epsilon = 8
    dataset_name = 'mnli'
    model_name = 'gpt2-large'
    dp = True ### can be T/F


    #tuning_method = 'prompt-tuning' ### can be 'prefix-tuning' or 'prompt-tuning'
    tuning_method = 'prefix-tuning' ### can be 'prefix-tuning' or 'prompt-tuning'
    for model_name in ['gpt2','gpt2-medium','gpt2-large','gpt2-xl']:
        for dataset_name in ['qnli','sst2','mnli']:
            for dp in [True,False]:

                    dp_str = 'dp' if dp else 'non_dp'
                    config_str = f"tuning_method-{tuning_method}-lr-{lr}-batch_size-{batch_size}-n_accumulation_steps-{n_accumulation_steps}-freeze_embedding-{freeze_embedding}-epochs-{epochs}-target_epsilon-{target_epsilon}"
                    save_path = os.path.join(BASE_DIR, 'checkpoints_gpt_all',config_str, dataset_name + '_' + model_name +  '_' + tuning_method + '_' + dp_str)


                    #### load victim model and tokenizer
                    adapter_name = get_adapter_name(tuning_method)
                    model = AutoModelForCausalLM.from_pretrained(model_name)
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = PeftModel.from_pretrained(model=model, model_id=os.path.join(save_path, adapter_name), adapter_name=adapter_name)
                    attacker = GEIA_attacker_gpt2_peft(model, tokenizer, attacker_config,config_str,
                                model_name, dp_str, need_proj = True, 
                                dataset_name = dataset_name, tuning_method = tuning_method)
                    ### add wanbd

                    name_=f"{dataset_name}-{model_name}-{dp_str}"
                    name_ = name_ + '-' + config_str
                    run = wandb.init(
                    # set the wandb project where this run will be logged
                    project=f"llm-atk-EIA-GPT-{dataset_name}",
                    name=name_,
                    reinit = True,
                    # track hyperparameters and run metadata
                    config=attacker_config)

                    attacker.train_attcker()
                    attacker.EIA_evaluate()
                    run.finish()