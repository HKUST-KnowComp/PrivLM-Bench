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
from transformers import AutoModel, AutoTokenizer,T5Tokenizer, T5EncoderModel, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
from GEIA_attacker import GEIA_base_attacker
import argparse


class T5_Dataset(Dataset):
    def __init__(self, data_list):        
        t5_data_list = []
        for data in data_list:
            data_split = data.split(' <SEP> ')
            t5_data = []
            t5_data.append(' <SEP> '.join(data_split[:-1]))
            t5_data.append(data_split[-1][-1] + ' </s>')
            t5_data_list.append(t5_data)
        self.data_list = t5_data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]
    

### generative DP_trainers
class GEIA_attacker_t5(GEIA_base_attacker):
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
        
        self.save_path = os.path.join(BASE_DIR, 'checkpoints-t5',config_str,
                                    dataset_name + '_' + model_name + '_' + tuning_method + '_' + dp_str + '_attacker')
        self.projection_save_path = os.path.join(BASE_DIR, 'checkpoints-t5',config_str,
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
            dummy_label = ['-1'] * len(batch_text)
            label = tokenizer(dummy_label, return_tensors='pt', padding=True, truncation=True)
            label_ids = label['input_ids'].to(self.device)  # tensors of input ids
            input_ids = inputs['input_ids']  # tensors of input ids
            input_ids = input_ids.to(self.device)
            #output = self.model(input_ids = input_ids,output_hidden_states=True, return_dict=True)
            output = self.model(input_ids = input_ids,labels=label_ids, return_dict=True)
            #hidden = output.encoder_last_hidden_state  # tuple of hidden states
            last_hidden_state = output.encoder_last_hidden_state  # tensor of shape (batch_size, seq_len, hidden_size)
            batch_embedding = self.get_embedding_from_hidden(last_hidden_state, mean_pooling=True)  # tensor of shape (batch_size, hidden_size)
        return batch_embedding  ##return (batch_size, hidden_size)


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


    for model_name in ['t5-base','t5-large','t5-xl']:
        for dataset_name in ['mnli']:
            for dp in [True,False]:
                dp_str = 'dp' if dp else 'non_dp'
                config_str = f"tuning_method-{tuning_method}-lr-{lr}-batch_size-{batch_size}-n_accumulation_steps-{n_accumulation_steps}-freeze_embedding-{freeze_embedding}-epochs-{epochs}-target_epsilon-{target_epsilon}"
                save_path = os.path.join(BASE_DIR, 'checkpoints-t5',config_str, dataset_name + '_' + model_name +  '_' + dp_str)


                #### load victim model and tokenizer
                model = T5ForConditionalGeneration.from_pretrained(save_path)
                if model_name == 't5-xl':
                    tokenizer = AutoTokenizer.from_pretrained('t5-3b')
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)

                attacker = GEIA_attacker_t5(model, tokenizer, attacker_config,config_str,
                            model_name, dp_str, need_proj = True, 
                            dataset_name = dataset_name, tuning_method = tuning_method)
                ### add wanbd

                name_=f"{dataset_name}-{model_name}-{dp_str}"
                name_ = name_ + '-' + config_str
                run = wandb.init(
                # set the wandb project where this run will be logged
                project=f"llm-atk-EIA-T5-{dataset_name}",
                name=name_,
                reinit = True,
                # track hyperparameters and run metadata
                config=attacker_config)

                attacker.train_attcker()
                attacker.EIA_evaluate()
                run.finish()