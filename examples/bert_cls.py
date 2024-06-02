'''
This version of BERT is based on seqclassification.
'''

import wandb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
#parent_dir = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)


import torch, transformers
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from private_transformers import PrivacyEngine
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dataset.data_process import get_sent_list_cls
import numpy as np
from transformers import AutoModel, AutoTokenizer
from eval.DEA.exposure_metric import calculate_single_exposure
#### example include BertForSequenceClassification ("bert-base-cased")
from transformers import AutoModelForCausalLM,AutoModelForSequenceClassification
import argparse
import config
from training_interface import DP_trainer
from tqdm import tqdm
from pprint import pprint

#MAX_SEQ_LEN = 512

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

## collate for batch processing
def collate(sent_a, sent_b, label, tokenizer):
    #encoding = tok(sentence_a, sentence_b, padding=True, truncation=True)
    # ignore the sentence_b for -100 (None)
    if sent_b[0] == -100:
        inputs = tokenizer(sent_a, padding=True, truncation=True, return_tensors="pt")
    else:
        inputs = tokenizer(sent_a,sent_b, padding=True, truncation=True, return_tensors="pt")
    return inputs, label

class BERT_CLS_tariner(DP_trainer):

    def get_model(self, model):
        if(self.num_labels):
            model = AutoModelForSequenceClassification.from_pretrained(model,num_labels=self.num_labels)
            print(f'load SequenceClassification model: {self.model_name}')
        else:
            raise NotImplementedError("num_labels is not given")
        return model
    def get_tokenizer(self, model):
        print(f'load tokenizer: {model}')
        return AutoTokenizer.from_pretrained(model)
    def load_model(self, model_path):
        del self.model
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    ### Get labels for corresponding dataset
    def get_dataloader(self, dataset_name):
        if dataset_name in ['qnli', "mnli", 'sst2']:
            sent_dict = get_sent_list_cls(dataset_name=dataset_name, data_type=self.data_type)
            data = CLS_dataset(sent_dict)
            self.num_labels = sent_dict["label_num"]
            print(f"load {dataset_name} dataset")

        else:
            raise NotImplementedError("Given dataset is not supported")
        if self.data_type == 'dev':
            return DataLoader(
                dataset=data,
                shuffle = False,
                batch_size=self.batch_size
                )
        return DataLoader(
                dataset=data,
                shuffle=True,
                batch_size=self.batch_size
            )
    
    def train_our_model(self):
        if self.dp:
            self.privacy_engine.attach(self.optimizer)
            print("privacy engine is on")
        self.model.to(self.device)
        print(f"LM: {self.model_name} is loaded")
        for epoch in range(self.epochs):
            ##### training code #####
            self.model.train()
            for idx, (sent_a, sent_b, label) in enumerate(tqdm(self.dataloader)):
                inputs, label = collate(sent_a, sent_b, label, self.tokenizer)
                inputs = {key: inputs[key].to(self.device) for key in inputs}
                label = torch.tensor(label).to(self.device)
                outputs = self.model(**inputs, labels=label,return_dict=True)
                ### not per sample loss
                logits = outputs.logits
                loss = F.cross_entropy(logits, label, reduction="none")
                if self.dp: # new added
                    if ((idx + 1) % self.n_accumulation_steps == 0) or ((idx + 1) == len(self.dataloader)):
                        self.optimizer.step(loss=loss)
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                    else:
                        print("virtual step")
                        self.optimizer.virtual_step(loss=loss)
                else:
                    loss = outputs.loss
                    loss.backward()
                    ### donghao added for gradient accumulation (same as virtual step)
                    if ((idx + 1) % self.n_accumulation_steps == 0) or ((idx + 1) == len(self.dataloader)):
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                    #self.optimizer.step()
                wandb.log({"batch loss": loss.mean(), "built-in loss":outputs.loss})
                #print(f"{epoch + 1}th epoch{idx}th batch output loss: {outputs.loss}. CE loss: {torch.mean(loss)}")
            # save checkpoints
            save_path = self.save_path
            self.model.save_pretrained(save_path)
            self.utility_evaluate()

            ##### training code #####

    def utility_evaluate(self):
        self.data_type = 'dev'
        self.eval_dataloader = self.get_dataloader(self.dataset_name)
        self.model.eval()
        self.model.to(self.device)
        right_count = 0
        for idx, (sent_a, sent_b, labels) in enumerate(tqdm(self.eval_dataloader)):
            inputs, labels = collate(sent_a, sent_b, labels, self.tokenizer)
            inputs = {key: inputs[key].to(self.device) for key in inputs}
            labels = torch.tensor(labels).to(self.device)
            outputs = self.model(**inputs, labels=labels,return_dict=True)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            right_count += torch.sum(preds == labels).item()
        acc = right_count / len(self.eval_dataloader.dataset)
        print(f"Accuracy: {acc}")
        wandb.log({"Validation Accuracy": acc})






if __name__ == "__main__":
    config = config.config
    datasets = ['qnli', 'sst2','mnli']
    models = ['bert-base-uncased', 'bert-large-uncased','roberta-base','roberta-large']
    for d in datasets:
        for m in models:
            for dp in [True, False]:


                config['dp'] = dp
                config['model'] = m
                config['dataset_name'] = d



                print("Config is loaded")
                pprint(config)
                trainer = BERT_CLS_tariner(**config)

                ### add wanbd
                dp_str = "dp" if config["dp"]  else "non_dp"
                name_=f"{config['dataset_name']}-{config['model']}-{dp_str}"
                config_str = trainer.config_str
                name_ = name_ + '-' + config_str
                run = wandb.init(
                # set the wandb project where this run will be logged
                project=f"eval0918-bert-cls-{config['dataset_name']}",
                name=name_,
                reinit = True,
                # track hyperparameters and run metadata
                config=config)

                #trainer.train_our_model()
                save_path = trainer.save_path
                trainer.load_model(save_path)
                print(f'load model from: {save_path}')
                trainer.utility_evaluate()
                run.finish()

    #trainer.canary_evaluate([100])

    #### eval
    #save_path = trainer.save_path
    #save_path
    #trainer.load_model(save_path)
    #trainer.utility_evaluate()