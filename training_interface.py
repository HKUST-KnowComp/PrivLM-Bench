import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import wandb
import torch, transformers
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from private_transformers import PrivacyEngine
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dataset.canary_dataset import CanaryDataset, EvaluationDataset
from utils import SequenceCrossEntropyLoss, calculate_perplexity_for_gpt, calculate_exposures
from dataset.canary_dataset import generate_evaluation_sequence_for_gpt
import numpy as np
from transformers import AutoModel, AutoTokenizer
from eval.DEA.exposure_metric import calculate_single_exposure
from transformers import AutoModelForCausalLM
import argparse
import config
from tqdm import tqdm
from eval.utility.utility_evaluation import utility_evaluation
import matplotlib.pyplot as plt


MAX_SEQ_LEN = 1024


### generative DP_trainers
class DP_trainer(object):
    def __init__(self,
                 model="distilgpt2",
                 optimizer="adam",
                 lr=1e-3,
                 batch_size=100,
                 n_accumulation_steps=1,
                 #sample_size=50000,
                 epochs=1,
                 max_grad_norm=0.1,
                 dp=True,
                 target_epsilon=3,
                 target_delta=1e-5,
                 clipping_mode='default',
                 dataset_name="personachat",
                 ### canary
                 canary_type_list=None,
                 insert_proportion_list=None,
                 insert_time_base_list=None,
                 tokenizer_len=None,
                 data_type="train",
                 #### scheduler args
                 warmup_steps = 40,
                 freeze_embedding = False,
                 ### evaluation
                 eva_method='logits',
                 num_decode_virtual_tokens=0,
                 num_beams=20,
                 generation_max_length=20,
                 num_return_sequence=10,
                 **kwargs,
                 ):
        self.warmup_steps = warmup_steps
        self.n_accumulation_steps = n_accumulation_steps # new added
        print("batch_size: ", batch_size, "n_accumulation_steps: ", n_accumulation_steps)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dp = dp
        self.data_type = data_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.freeze_embedding = freeze_embedding
        self.n_accumulation_steps = n_accumulation_steps
        self.lr = lr
        self.target_epsilon = target_epsilon
        #self.virtual_batch_size = virtual_batch_size
        
        ##### canary
        self.canary_type_list = canary_type_list
        self.insert_proportion_list = insert_proportion_list
        self.insert_time_base_list = insert_time_base_list

        self.dataset_name = dataset_name
        self.dataloader = self.get_dataloader(dataset_name)
        self.tokenizer = self.get_tokenizer(model)

        self.tokenizer_len = tokenizer_len
        self.model_name = model

        self.model = self.get_model(model)
        self.freeze_embedding = freeze_embedding
        if freeze_embedding:
            self.freeze_model_embedding()
        self.optimizer = self.get_optimizer(optimizer, lr)
        self.lr_scheduler = self.get_scheduler()
        #### evaluation
        self.eva_method = eva_method
        self.num_decode_virtual_tokens = num_decode_virtual_tokens
        self.num_beams = num_beams
        self.generation_max_length = generation_max_length
        self.num_return_sequence = num_return_sequence

        dp_str = "dp" if self.dp else "non_dp"
        self.tuning_method = 'finetune'
        self.config_str = f"tuning_method-{self.tuning_method}-lr-{self.lr}-batch_size-{self.batch_size}-n_accumulation_steps-{self.n_accumulation_steps}-freeze_embedding-{self.freeze_embedding}-epochs-{self.epochs}-target_epsilon-{self.target_epsilon}"
        if len(self.canary_type_list) > 0:
            self.config_str = "canary_inserted-" + self.config_str
        self.save_path = os.path.join(BASE_DIR, 'checkpoints',self.config_str, self.dataset_name + '_' + self.model_name + '_' + dp_str)
        if self.dp:
            self.privacy_engine = PrivacyEngine(
                self.model,
                batch_size=batch_size * n_accumulation_steps, # new added batch size for dp account is batch_size * n_accumulation_steps
                sample_size=len(self.dataloader.dataset),
                epochs=epochs,
                max_grad_norm=max_grad_norm,
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                clipping_mode=clipping_mode,
                ## hr: this flag can bypass the model constraints, make sure to implement carefully
                skip_checks=True,
            )


    def get_scheduler(self):
        #self.gradient_accumulation_steps
        t_total = int(len(self.dataloader) // self.n_accumulation_steps * self.epochs)
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=t_total
        )

    def freeze_model_embedding(self):
        self.model.get_input_embeddings().requires_grad_(False)
        print("embedding layer is frozen")

    def get_tokenizer(self, model):
        return AutoTokenizer.from_pretrained(model)

    def get_model(self, model):
        model = AutoModelForCausalLM.from_pretrained(model)
        return model

    def load_model(self, model_path):
        del self.model
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

    def get_optimizer(self, optimizer, lr):
        if optimizer == "adam":
            optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        elif optimizer == "adamw":
            optimizer = AdamW(params=self.model.parameters(), lr=lr)
        else:
            raise NotImplementedError("Given optimizer type is not s upported")
        return optimizer

    def get_dataloader(self, dataset_name):
        if dataset_name in ['personachat', 'qnli', "mnli", 'sst2']:
            if self.data_type == 'train':
                data = CanaryDataset(self.canary_type_list, self.insert_proportion_list, self.insert_time_base_list,
                                     dataset_name=dataset_name, data_type=self.data_type)
            else:
                data = EvaluationDataset(dataset_name=dataset_name, data_type=self.data_type)
        else:
            raise NotImplementedError("Given dataset is not supported")
        if self.data_type == 'dev':
            return DataLoader(
                dataset=data,
                shuffle=False,
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
        # self.model.train()
        self.model.to(self.device)
        print(f"LM: {self.model_name} is loaded")
        criterion = SequenceCrossEntropyLoss()
        print("Begin training")
        for epoch in range(self.epochs):
            ##### training code #####
            self.model.train()
            for idx, batch_text in enumerate(tqdm(self.dataloader)):
                record_loss = self.train_on_batch(batch_text=batch_text, model=self.model,
                                                              tokenizer=self.tokenizer, criterion=criterion,
                                                              device=self.device)
                if self.dp: # new added for gradient accumulation
                    if ((idx + 1) % self.n_accumulation_steps == 0) or ((idx + 1) == len(self.dataloader)):
                        self.optimizer.step(loss=record_loss)
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                    else:
                        self.optimizer.virtual_step(loss=record_loss)
                else:
                    record_loss.backward()
                    if ((idx + 1) % self.n_accumulation_steps == 0) or ((idx + 1) == len(self.dataloader)):
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                #print(f"{epoch + 1}th epoch, {idx}th batch: output loss: {torch.mean(record_loss)}")
            # save checkpoints
            self.save_checkpoints()
            self.utility_evaluate()
            ##### training code #####
    def save_checkpoints(self):
        save_path = self.save_path
        self.model.save_pretrained(save_path)
    
    def train_on_batch(self, batch_text, model, tokenizer, criterion, device):
        tokenizer.pad_token = tokenizer.eos_token
        if(self.tokenizer_len):
            inputs = tokenizer(batch_text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.tokenizer_len)
        else:
            inputs = tokenizer(batch_text, return_tensors='pt', padding=True, truncation=True)

        input_ids = inputs['input_ids']  # tensors of input ids
        ### GPT-2 tokenizer requires extra <eos> padding for batch training (no eos in the end)
        if('gpt' in self.model_name and not self.tokenizer_len and input_ids.shape[1] < MAX_SEQ_LEN):
            pad_id = tokenizer.encode(tokenizer.pad_token)[0]
            pad_pt = torch.ones((input_ids.shape[0],1),dtype=input_ids.dtype) * pad_id
            input_ids = torch.cat((input_ids,pad_pt),dim=1)
        input_ids = input_ids.to(device)
        labels = input_ids.clone()
        past = None
        logits, past = model(input_ids = input_ids, past_key_values=past, return_dict=False)
        logits = logits[:, :-1].contiguous()
        target = labels[:, 1:]
        target_mask = torch.ones_like(target).float()
        if self.dp:
            loss = criterion(logits, target, target_mask, label_smoothing=0.02, reduce=True)
        else:
            loss = criterion(logits, target, target_mask, label_smoothing=0.02, reduce="batch")
        return loss

    def canary_evaluate(self, use_full_text=True):
        for canary_type, insert_proportion, insert_time_base in zip(self.canary_type_list,
                                                                    self.insert_proportion_list,
                                                                    self.insert_time_base_list
                                                                    ):
            insert, not_insert = generate_evaluation_sequence_for_gpt(canary_type=canary_type, use_full_text=use_full_text,
                                                              insert_proportion=insert_proportion)
            insert_seqs_perplexity = calculate_perplexity_for_gpt(insert, self.model, self.tokenizer,
                                                          num_decode_virtual_tokens=self.num_decode_virtual_tokens,
                                                          tuning_method=self.tuning_method,
                                                          device=self.device)
            not_insert_seqs_perplexity = calculate_perplexity_for_gpt(not_insert, self.model, self.tokenizer,
                                                              num_decode_virtual_tokens=self.num_decode_virtual_tokens,
                                                              tuning_method=self.tuning_method,
                                                              device=self.device)
            exposures = calculate_exposures(insert_seqs_perplexity, not_insert_seqs_perplexity)
            insert_time_list = [i * insert_time_base * self.epochs for i in range(1, len(insert) + 1)]
            exposures = exposures.tolist()

            plt.figure(figsize=(15, 10), dpi=80)
            plt.plot(insert_time_list, exposures)

            plt.xlabel('Number of insertions')
            plt.ylabel('Exposure')
            plt.title(f'The exposures of canary type {canary_type}')
            plt.show()

            plt.savefig(os.path.join(BASE_DIR, "canary_figs", f"{self.tuning_method}", f"{canary_type}.png"))


    def utility_evaluate(self):
        self.data_type = 'dev'
        self.eval_dataloader = self.get_dataloader(self.dataset_name)
        self.model.eval()
        self.model.to(self.device)
        acc = utility_evaluation(model=self.model,tokenizer=self.tokenizer,dataloader=self.eval_dataloader,
                                 dataset_name=self.dataset_name,
                                 eva_method=self.eva_method,
                                 device=self.device,
                                 num_virtual_token=self.num_decode_virtual_tokens,
                                 num_beams=self.num_beams,
                                 num_return_sequences=self.num_return_sequence,
                                 max_length=self.generation_max_length
                                 )
        print(f"acc = {acc}")

if __name__ == "__main__":
    config = config.config

    trainer = DP_trainer(**config)
    trainer.train_our_model()
    trainer.utility_evaluate()
    trainer.canary_evaluate()
