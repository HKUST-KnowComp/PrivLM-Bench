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
from transformers import AutoModelForMaskedLM
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import argparse
from dataset.data_process import get_sent_list_cls
from GEIA_attacker import GEIA_base_attacker
from decode_utils import eval_on_batch
import json
from decode_eval import eval_eia


MAX_SEQ_LEN = 256
## collate for batch processing
def collate(sent_a, sent_b, label, tokenizer):
    #encoding = tok(sentence_a, sentence_b, padding=True, truncation=True)
    # ignore the sentence_b for -100 (None)
    if sent_b[0] == -100:
        inputs = tokenizer(sent_a, padding=True, truncation=True, return_tensors="pt")
        batch_text = sent_a
    else:
        inputs = tokenizer(sent_a,sent_b, padding=True, truncation=True, return_tensors="pt")
        batch_text = [sent_a[i]+sent_b[i] for i in range(len(sent_a))]
    return inputs, label,batch_text


class CLS_dataset(Dataset):
    def __init__(self, sent_dict): 
        self.sent_a = sent_dict["sent_a"]
        self.sent_b = sent_dict["sent_b"]
        self.label = sent_dict["label"]
        self.num_labels = sent_dict["label_num"]
        assert len(self.sent_a) == len(self.label)

        

    def __getitem__(self, index): 
        sent_a = self.sent_a[index]
        label = self.label[index]
        if self.sent_b is None:
            return sent_a, -100, label
        sent_b = self.sent_b[index]
        return sent_a, sent_b, label

    def __len__(self): 
        return len(self.sent_a)
    

class GEIA_bert_attacker(GEIA_base_attacker):

    def get_dataloader(self, dataset_name, data_type = 'train'):
        if dataset_name in ['qnli', "mnli", 'sst2']:
            if data_type == 'train':
                sent_dict = get_sent_list_cls(dataset_name=dataset_name, data_type=data_type,is_aux=True)
            elif data_type == 'dev':
                sent_dict = get_sent_list_cls(dataset_name=dataset_name, data_type=data_type,return_all=True)
            data = CLS_dataset(sent_dict)
            self.num_labels = sent_dict["label_num"]
            print(f"load {dataset_name} dataset")

        else:
            raise NotImplementedError("Given dataset is not supported")
        return DataLoader(
                dataset=data,
                shuffle=True,
                batch_size=self.batch_size
            )


    def get_batch_embedding(self, inputs):
        with torch.no_grad():
            inputs = {key: inputs[key].to(self.device) for key in inputs}
            outputs = self.model(**inputs,return_dict=True)
            last_hidden_state = outputs.last_hidden_state
            batch_embedding = self.get_embedding_from_hidden(last_hidden_state, mean_pooling=True)
        return batch_embedding  ##return (batch_size, hidden_size)

    def train_attcker(self):
        criterion = self.criterion
        for epoch in range(self.epochs):
            self.model.eval()
            self.attacker_model.train()
            for idx, (sent_a, sent_b, label) in enumerate(tqdm(self.dataloader)):
                inputs, label, batch_text = collate(sent_a, sent_b, label, self.tokenizer)
                if(inputs['input_ids'].shape[1]  + 1> MAX_SEQ_LEN):
                    continue

                batch_embedding = self.get_batch_embedding(inputs = inputs)
                if self.need_porj:
                    batch_embedding = self.projection(batch_embedding)

                model = self.attacker_model
                tokenizer = self.attacker_tokenizer
                tokenizer.pad_token = tokenizer.eos_token
                if(self.tokenizer_len):
                    inputs = tokenizer(batch_text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.tokenizer_len)
                else:
                    inputs = tokenizer(batch_text, return_tensors='pt', padding=True, truncation=True)

                input_ids = inputs['input_ids']  # tensors of input ids
                input_ids = input_ids.to(self.device)
                labels = input_ids.clone()
                # embed the input ids using GPT-2 embedding
                input_emb = model.transformer.wte(input_ids)
                # add extra dim to cat together
                batch_X = batch_embedding
                assert len(batch_X.size()) == 2
                batch_X_unsqueeze = torch.unsqueeze(batch_X, 1)
                inputs_embeds = torch.cat((batch_X_unsqueeze,input_emb),dim=1)   #[batch,max_length+1,emb_dim (1024)]
                # need to move to device later
                inputs_embeds = inputs_embeds
                output = model(inputs_embeds=inputs_embeds,past_key_values  = None,return_dict=True)
                logits = output.logits
                logits = logits[:, :-1].contiguous()
                target = labels.contiguous()
                target_mask = torch.ones_like(target).float()
                loss = criterion(logits, target, target_mask, label_smoothing=0.02, reduce="batch")   

                record_loss = loss.item()
                perplexity = np.exp(record_loss)

                ### update params
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                wandb.log({"batch loss": record_loss, "perplexity":perplexity})
            ### save
            self.attacker_model.save_pretrained(self.save_path)
            if self.need_porj:
                torch.save(self.projection.state_dict(), self.projection_save_path)

    def EIA_evaluate(self):
        self.model.eval()
        self.attacker_model.eval()
        self.eval_dataloader = self.get_dataloader(dataset_name, data_type='dev')
        decode_config = self.prepare_decode_config()
        sent_dict = {}
        sent_dict['pred'] = []
        sent_dict['gt'] = []
        with torch.no_grad():  
            for idx, (sent_a, sent_b, label) in enumerate(tqdm(self.eval_dataloader)):
                inputs, label, batch_text = collate(sent_a, sent_b, label, self.tokenizer)   
                if(inputs['input_ids'].shape[1] + 1> MAX_SEQ_LEN):
                    continue             
                batch_embedding = self.get_batch_embedding(inputs = inputs)
                if self.need_porj:
                    batch_embedding = self.projection(batch_embedding)
                model = self.attacker_model
                tokenizer = self.attacker_tokenizer
                tokenizer.pad_token = tokenizer.eos_token
                sent_list, gt_list = eval_on_batch(batch_X=batch_embedding,batch_D=batch_text,model=model,tokenizer=tokenizer,device=self.device,config=decode_config)
                print(f'testing {idx} batch done with {(idx+1)*batch_size} samples')
                sent_dict['pred'].extend(sent_list)
                sent_dict['gt'].extend(gt_list) 
                #break
            generation_save_path = os.path.join(self.save_path,'eia_generation.json')
            with open(generation_save_path, 'w') as f:
                json.dump(sent_dict, f,indent=4)

            eval_eia(sent_dict,self.save_path)
            return sent_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training external NN as baselines')
    parser.add_argument('--porj_path', type=str, default='/home/data/hlibt/p_bench', help='project path')
    args = parser.parse_args()

    project_path = args.porj_path
    tuning_method = 'mlm'
    lr = 1e-4
    batch_size = 4
    n_accumulation_steps = 256
    freeze_embedding = 'True'
    epochs = 5
    target_epsilon = 8
    #dataset_name = 'mnli'
    #model_name = 'roberta-large'
    #dp = False ### can be T/F
    for d in ['qnli','sst2','mnli']:
        dataset_name = d
        #for model_name in ['roberta-large','roberta-base']:
        for model_name in ['bert-base-uncased','roberta-base','roberta-large','bert-large-uncased']:
            for dp in [True,False]:
                dp_str = 'dp' if dp else 'non_dp'
                config_str = f"tuning_method-{tuning_method}-lr-{lr}-batch_size-{batch_size}-n_accumulation_steps-{n_accumulation_steps}-freeze_embedding-{freeze_embedding}-epochs-{epochs}-target_epsilon-{target_epsilon}"
                if tuning_method == 'finetune':
                    save_path = os.path.join(BASE_DIR, 'checkpoints',config_str, dataset_name + '_' + model_name + '_' + dp_str)
                else:
                    save_path = os.path.join(BASE_DIR, 'checkpoints',config_str, dataset_name + '_' + model_name + '_' + tuning_method + '_' + dp_str)


                #### load victim model and tokenizer

                model = AutoModelForMaskedLM.from_pretrained(save_path)
                ### get cls base model
                model = model.base_model
                tokenizer = AutoTokenizer.from_pretrained(model_name)

                attacker = GEIA_bert_attacker(model, tokenizer, attacker_config,config_str,
                            model_name, dp_str, need_proj = True, 
                            dataset_name = dataset_name, tuning_method = tuning_method)
                ### add wanbd

                name_=f"{dataset_name}-{model_name}-{dp_str}"
                name_ = name_ + '-' + config_str
                run = wandb.init(
                # set the wandb project where this run will be logged
                project=f"eval0921-llm-atk-EIA-{dataset_name}",
                name=name_,
                reinit = True,
                # track hyperparameters and run metadata
                config=attacker_config)


                #attacker.train_attcker()
                #attacker.EIA_evaluate()

                #### load pretrain attacker for eval
                gpt_path = attacker.save_path
                projection_path = attacker.projection_save_path
                attacker.load_attacker_from_path(gpt_path,projection_path)
                attacker.EIA_evaluate()
                run.finish()