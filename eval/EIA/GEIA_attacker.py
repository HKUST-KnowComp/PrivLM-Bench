'''
Attacker implementation from the paper
Sentence Embedding Leaks More Information than You Expect: Generative Embedding Inversion Attack to Recover the Whole Sentence
https://arxiv.org/pdf/2305.03010.pdf
'''

import wandb
import os
import sys
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
from transformers import AdamW,get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from utils import SequenceCrossEntropyLoss
from tqdm import tqdm
import numpy as np
import argparse
from dataset.data_process import get_sent_list
from decode_utils import eval_on_batch
import json
from decode_eval import eval_eia

class linear_projection(nn.Module):
    def __init__(self, in_num, out_num=1024):
        super(linear_projection, self).__init__()
        self.fc1 = nn.Linear(in_num, out_num)

    def forward(self, x, use_final_hidden_only = True):
        # x should be of shape (?,in_num) according to gpt2 output
        out_shape = x.size()[-1]
        assert(x.size()[1] == out_shape)
        out = self.fc1(x)


        return out


class CLM_dataset(Dataset):
    def __init__(self, sent_list): 
        self.sent_list = sent_list

        

    def __getitem__(self, index): 
        sent = self.sent_list[index]
        return sent

    def __len__(self): 
        return len(self.sent_list)
    


### generative DP_trainers
class GEIA_base_attacker(object):
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
        self.model_name = model_name
        self.tuning_method = tuning_method
        self.device = attacker_config['device']
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        self.attacker_config = attacker_config
        self.need_proj = need_proj
        self.warmup_steps = attacker_config['warmup_steps']
        self.epochs = attacker_config['epochs']
        self.batch_size = attacker_config['batch_size']
        self.tokenizer_len = attacker_config['tokenizer_len']
        self.need_porj = need_proj
        self.num_virtual_tokens = num_virtual_tokens
        ### overwrite this for save and load
        self.n_accumulation_steps = 1
        self.tuning_method = tuning_method
        self.save_path = os.path.join(BASE_DIR, 'checkpoints',config_str,
                                    dataset_name + '_' + model_name + '_' + tuning_method + '_' + dp_str + '_attacker')
        self.projection_save_path = os.path.join(BASE_DIR, 'checkpoints',config_str,
                                    dataset_name + '_' + model_name + '_' + tuning_method + '_' + dp_str + '_attacker_projection')
        ### Init for attackers (For simplicity, we use the GPT-2 model as the attacker model)
        self.attacker_model = AutoModelForCausalLM.from_pretrained(attacker_config['model_dir']).to(self.device)
        self.attacker_tokenizer = AutoTokenizer.from_pretrained(attacker_config['model_dir'])
        param_optimizer = list(self.attacker_model.named_parameters())
        no_decay = ['bias', 'ln', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, 
                  lr=3e-5,
                  eps=1e-06)
        
        ### extra projection
        if need_proj:
            self.projection = self.get_projection().to(self.device)
            self.optimizer.add_param_group({'params': self.projection .parameters()})
        else:
            self.projection = None

        self.dataset_name = dataset_name
        self.dataloader = self.get_dataloader(dataset_name)
        self.criterion = SequenceCrossEntropyLoss()
        self.lr_scheduler = self.get_scheduler()
        print('Embedding Inversion Init done')

    def get_projection(self):
        ### get in_num and out_num
        def get_emb_dim(model):
            #### GPT2LMHeadModel:  model.transformer.wte.weight
            #### BERT: model.embeddings.word_embeddings.weight
            if hasattr(model, 'transformer'):
                embedding_matrix = model.transformer.wte.weight
            elif hasattr(model, 'shared'):
                embedding_matrix = model.shared.weight
            ### T5ForConditionalGeneration
            elif hasattr(model, 'encoder') and hasattr(model, 'decoder'):
                print('get projection from T5ForConditionalGeneration')
                embedding_matrix = model.shared.weight
            elif hasattr(model, 'embeddings'):
                embedding_matrix = model.embeddings.word_embeddings.weight
            elif hasattr(model, 'bert'):
                embedding_matrix = model.bert.embeddings.word_embeddings.weight
            elif hasattr(model, 'roberta'):
                embedding_matrix = model.roberta.embeddings.word_embeddings.weight
            else:
                raise ValueError('Cannot find embedding matrix')
            print(f'embedding dimension: {embedding_matrix.size()}')
            return embedding_matrix.size()[-1]
        in_num = get_emb_dim(self.model)
        out_num = get_emb_dim(self.attacker_model)
        projection = linear_projection(in_num=in_num, out_num=out_num)
        return projection


    def get_dataloader(self, dataset_name, data_type = 'train'):
        assert data_type in ['train', 'dev']
        if dataset_name in ['qnli', "mnli", 'sst2']:
            if data_type == 'train':
                sent_dict = get_sent_list(dataset_name=dataset_name, data_type=data_type,is_aux=True)
            elif data_type == 'dev':
                sent_dict = get_sent_list(dataset_name=dataset_name, data_type=data_type,return_all=True)
            data = CLM_dataset(sent_dict)
            #self.num_labels = sent_dict["label_num"]
            print(f"load {dataset_name} dataset")

        else:
            raise NotImplementedError("Given dataset is not supported")
        return DataLoader(
                dataset=data,
                shuffle=True,
                batch_size=self.batch_size
            )
    
    def get_scheduler(self):
        #self.gradient_accumulation_steps
        t_total = int(len(self.dataloader) // self.n_accumulation_steps * self.epochs)
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=t_total
        )
    def train_attcker(self):
        criterion = self.criterion
        for epoch in range(self.epochs):
            self.model.eval()
            self.attacker_model.train()
            for idx, batch_text in enumerate(tqdm(self.dataloader)):
                batch_embedding = self.get_batch_embedding(batch_text)
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

                #wandb.log({"batch loss": record_loss, "perplexity":perplexity})
            ### save
            self.attacker_model.save_pretrained(self.save_path)
            if self.need_porj:
                torch.save(self.projection.state_dict(), self.projection_save_path)
            


    '''
    Perform sentence generation using the victim embedding
    '''
    def prepare_decode_config(self):
        config = {}
        config['model'] = self.attacker_model
        config['tokenizer'] = self.attacker_tokenizer
        config['decode'] = 'beam'       ### beam or greedy
        config['use_opt'] = False

        return config
    def EIA_evaluate(self):
        self.model.eval()
        self.attacker_model.eval()
        #### LHR: Make sure your eval dataloader works
        self.eval_dataloader = self.get_dataloader(self.dataset_name, data_type='dev')
        decode_config = self.prepare_decode_config()
        sent_dict = {}
        sent_dict['pred'] = []
        sent_dict['gt'] = []
        with torch.no_grad():  
            for idx, batch_text in enumerate(tqdm(self.eval_dataloader)):
                batch_embedding = self.get_batch_embedding(batch_text)
                if self.need_porj:
                    batch_embedding = self.projection(batch_embedding)

                model = self.attacker_model
                tokenizer = self.attacker_tokenizer
                tokenizer.pad_token = tokenizer.eos_token

                sent_list, gt_list = eval_on_batch(batch_X=batch_embedding,batch_D=batch_text,model=model,tokenizer=tokenizer,device=self.device,config=decode_config)
                print(f'testing {idx} batch done with {idx*self.batch_size} samples')
                sent_dict['pred'].extend(sent_list)
                sent_dict['gt'].extend(gt_list) 
            generation_save_path = os.path.join(self.save_path,'eia_generation.json')
            with open(generation_save_path, 'w') as f:
                json.dump(sent_dict, f,indent=4)

            eval_eia(sent_dict,self.save_path)
            return sent_dict



                


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
        
    def get_embedding_from_hidden(self,hidden,mean_pooling=True):
        '''
        hidden: (batch_size, seq_len, hidden_size)
        if mean_pooling is True, use mean pooling to get embedding from hidden
        else use the first token embedding
        return (batch_size, hidden_size)
        '''
        if mean_pooling:
            return torch.mean(hidden, dim=1)
        else:
            return hidden[:,0,:]


    def load_attacker_from_path(self,gpt_path,projection_path):
        del self.attacker_model
        self.attacker_model = AutoModelForCausalLM.from_pretrained(gpt_path).to(self.device)
        if self.need_porj:
            self.projection.load_state_dict(torch.load(projection_path))
        print('Pretrained Attacker loaded')

    def load_attacker(self):
        del self.attacker_model
        if os.path.exists(self.save_path):
            self.attacker_model = AutoModelForCausalLM.from_pretrained(self.save_path).to(self.device)
            print('load attacker from default save path')
        else:
            raise ValueError('Cannot find saved attacker model')
        if self.need_porj:
            if os.path.exists(self.projection_save_path):
                self.projection.load_state_dict(torch.load(self.projection_save_path))
                print('load projection from default save path')
            else:
                raise ValueError('Cannot find saved projection model')




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
    model_name = 'gpt2'
    dp = True ### can be T/F
    dp_str = 'dp' if dp else 'non_dp'
    config_str = f"tuning_method-{tuning_method}-lr-{lr}-batch_size-{batch_size}-n_accumulation_steps-{n_accumulation_steps}-freeze_embedding-{freeze_embedding}-epochs-{epochs}-target_epsilon-{target_epsilon}"
    save_path = os.path.join(BASE_DIR, 'checkpoints_gpt',config_str, dataset_name + '_' + model_name + '_' + tuning_method + '_' + dp_str)


    #### load victim model and tokenizer
    victim_gpt_name = 'gpt2'
    model = AutoModelForCausalLM.from_pretrained(victim_gpt_name)
    tokenizer = AutoTokenizer.from_pretrained(victim_gpt_name)

    attacker = GEIA_base_attacker(model, tokenizer, attacker_config,config_str,
                model_name, dp_str, need_proj = True, 
                dataset_name = dataset_name, tuning_method = tuning_method)
    ### add wanbd

    name_=f"{dataset_name}-{model_name}-{dp_str}"
    name_ = name_ + '-' + config_str
    run = wandb.init(
    # set the wandb project where this run will be logged
    project=f"llm-atk-EIA-{dataset_name}",
    name=name_,
    reinit = True,
    # track hyperparameters and run metadata
    config=attacker_config)

    attacker.train_attcker()
    attacker.EIA_evaluate()
    run.finish()