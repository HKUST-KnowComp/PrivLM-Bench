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
from GEIA_attacker import GEIA_base_attacker


### generative DP_trainers
class GEIA_attacker_gpt2(GEIA_base_attacker):
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
        
        self.save_path = os.path.join(BASE_DIR, 'checkpoints_gpt_all',config_str,
                                    dataset_name + '_' + model_name + '_' + tuning_method + '_' + dp_str + '_attacker')
        self.projection_save_path = os.path.join(BASE_DIR, 'checkpoints_gpt_all',config_str,
                                    dataset_name + '_' + model_name + '_' + tuning_method + '_' + dp_str + '_attacker_projection')
        

    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training external NN as baselines')
    parser.add_argument('--porj_path', type=str, default='/home/data/hlibt/p_bench', help='project path')
    args = parser.parse_args()

    project_path = args.porj_path
    tuning_method = 'finetune'
    lr = 1e-4
    batch_size = 4
    n_accumulation_steps = 256
    freeze_embedding = 'True'
    epochs = 5
    target_epsilon = 8
    dataset_name = 'mnli'
    model_name = 'gpt2-large'
    dp = True ### can be T/F
    dp_str = 'dp' if dp else 'non_dp'

    #for model_name in ['gpt2','gpt2-medium','gpt2-large','gpt2-xl']:
    #for model_name in ['gpt2','gpt2-medium']:
    for model_name in ['gpt2-large','gpt2-xl']:
        for dataset_name in ['qnli','sst2','mnli']:
            for dp in [True,False]:

                dp_str = 'dp' if dp else 'non_dp'
                config_str = f"tuning_method-{tuning_method}-lr-{lr}-batch_size-{batch_size}-n_accumulation_steps-{n_accumulation_steps}-freeze_embedding-{freeze_embedding}-epochs-{epochs}-target_epsilon-{target_epsilon}"
                save_path = os.path.join(BASE_DIR, 'checkpoints_gpt_all',config_str, dataset_name + '_' + model_name +  '_' + dp_str)


                #### load victim model and tokenizer
                model = AutoModelForCausalLM.from_pretrained(save_path)
                tokenizer = AutoTokenizer.from_pretrained(model_name)

                attacker = GEIA_attacker_gpt2(model, tokenizer, attacker_config,config_str,
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