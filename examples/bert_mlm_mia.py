'''
This version of BERT is based on masked language modeling for text infilling optimization.
'''
import wandb
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# parent_dir = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

import torch, transformers
from private_transformers import PrivacyEngine
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dataset.data_process import get_sent_list,get_sent_list_ori_order
import numpy as np
from transformers import AutoModel, AutoTokenizer
from eval.DEA.exposure_metric import calculate_single_exposure
#### example include BertForSequenceClassification ("bert-base-cased")
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM
import argparse
import config
from training_interface import DP_trainer
from tqdm import tqdm
from pprint import pprint
import copy
# MAX_SEQ_LEN = 512

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='dataset name')
parser.add_argument('--batch_size', type=int, default=8, help='dataset name')
parser.add_argument('--model_name', type=str, default='bert-base-cased', help='model name')
parser.add_argument('--dataset_name', type=str, default='qnli', help='model name')
parser.add_argument('--dp', type=bool, default=False, help='whether to use dp')
args = parser.parse_args()


class INFILL_dataset(Dataset):
    def __init__(self, sent_list):
        self.sent_list = sent_list

    def __getitem__(self, index):
        sent = self.sent_list[index]
        return sent

    def __len__(self):
        return len(self.sent_list)


def collate(batch_sent, tokenizer, eval=False):
    # encoding = tok(sentence_a, sentence_b, padding=True, truncation=True)
    # ignore the sentence_b for -100 (None)
    mask_token = tokenizer.mask_token  # '[MASK]'
    mask_token_id = tokenizer.mask_token_id  # 103
    gt_inputs = tokenizer(batch_sent, padding=True, truncation=True, return_tensors="pt")
    labels = gt_inputs["input_ids"]
    numerical_labels = []
    for i, sent in enumerate(batch_sent):
        if sent[-1].isdigit() and sent[-2] == ' ':
            numerical_labels.append(int(sent[-1]))
            sent = sent[:-1] + mask_token
            batch_sent[i] = sent
        else:
            raise NotImplementedError("Infilling only supports digit masking as labels")
    # labels = copy.deepcopy(batch_sent)
    inputs = tokenizer(batch_sent, padding=True, truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"]
    # labels = inputs["input_ids"].clone()
    mask_token_indices = input_ids == mask_token_id
    labels[~mask_token_indices] = -100
    assert labels.shape == input_ids.shape
    # print('load batch data')

    if eval:
        index = labels > -100
        index = index.nonzero()  ### 2 d tensor for index the label position
        # labels[index[:,0],index[:,1]]   ###tensor([1014, 1015, 1016, 1014]) ###[0, 1, 2, 0]
        tokenized_labels = labels[index[:, 0], index[:, 1]]
        return inputs, tokenized_labels, index
    return inputs, labels


class BERT_INFILL_tariner(DP_trainer):
    def __init__(self, config, **kwargs):
        super().__init__(**config)
        dp_str = "dp" if self.dp else "non_dp"
        self.tuning_method = 'mlm'
        self.config_str = f"tuning_method-{self.tuning_method}-lr-{self.lr}-batch_size-{self.batch_size}-n_accumulation_steps-{self.n_accumulation_steps}-freeze_embedding-{self.freeze_embedding}-epochs-{self.epochs}-target_epsilon-{self.target_epsilon}"
        self.save_path = os.path.join(BASE_DIR, 'checkpoints', self.config_str,
                                      self.dataset_name + '_' + self.model_name + '_' + self.tuning_method + '_' + dp_str)

    def get_model(self, model):
        model = AutoModelForMaskedLM.from_pretrained(model)
        print(f'load MaskedLM model (for infilling): {self.model_name}')
        return model

    def get_tokenizer(self, model):
        print(f'load tokenizer: {model}')
        return AutoTokenizer.from_pretrained(model)

    def load_model(self, model_path):
        del self.model
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)

    ### Get labels for corresponding dataset
    def get_dataloader(self, dataset_name):
        if dataset_name in ['qnli', "mnli", 'sst2']:
            sent_dict = get_sent_list(dataset_name=dataset_name, data_type=self.data_type,SEED_NUMBER=args.seed)
            data = INFILL_dataset(sent_dict)
            # self.num_labels = sent_dict["label_num"]
            self.num_labels = 2 if (dataset_name == 'sst2' or dataset_name == 'qnli') else 3

            print(f"load {dataset_name} dataset")

        else:
            raise NotImplementedError("Given dataset is not supported")
        return DataLoader(
            dataset=data,
            shuffle=True,
            batch_size=self.batch_size
        )

    def get_dataloader_test_mia(self, dataset_name):
        if dataset_name in ['qnli', "mnli", 'sst2']:
            sent_dict,train_index,test_index = get_sent_list_ori_order(dataset_name=dataset_name, data_type=self.data_type)
            data = INFILL_dataset(sent_dict)
            #self.num_labels = sent_dict["label_num"]
            self.num_labels = 2 if (dataset_name == 'sst2' or dataset_name == 'qnli') else 3

            print(f"load {dataset_name} dataset")

        else:
            raise NotImplementedError("Given dataset is not supported")
        return DataLoader(
                dataset=data,
                shuffle=False,
                batch_size=self.batch_size
            ),train_index,test_index

    def train_our_model(self):
        if self.dp:
            self.privacy_engine.attach(self.optimizer)
            print("privacy engine is on")
        self.model.train()
        self.model.to(self.device)
        print(f"LM: {self.model_name} is loaded")
        for epoch in range(self.epochs):
            ##### training code #####
            self.model.train()
            for idx, (batch_sent) in enumerate(tqdm(self.dataloader)):
                inputs, label = collate(batch_sent, self.tokenizer)
                inputs = {key: inputs[key].to(self.device) for key in inputs}
                label = torch.tensor(label).to(self.device)
                outputs = self.model(**inputs, labels=label, return_dict=True)
                # shape : (batch, sequence_length, num_classes)
                logits = outputs.logits
                vocab_size = logits.shape[-1]
                seq_len = logits.shape[1]
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")  # -100 index = padding token
                # masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
                loss = loss_fct(logits.view(-1, vocab_size), label.view(-1))
                loss = loss.view(-1, seq_len)
                ### convert back to per sample loss of shape (batch )
                loss = loss.sum(1)
                if self.dp:  # new added
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
                wandb.init()
                wandb.log({"batch loss": loss.mean(), "built-in loss": outputs.loss})

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
        cand_list = self.get_candidate_list()
        cand_list = torch.tensor(cand_list).to(self.device)
        for idx, (batch_sent) in enumerate(tqdm(self.eval_dataloader)):
            inputs, tokenized_labels, index = collate(batch_sent, self.tokenizer, eval=True)
            inputs = {key: inputs[key].to(self.device) for key in inputs}
            label = torch.tensor(tokenized_labels).to(self.device)
            outputs = self.model(**inputs, return_dict=True)
            logits = outputs.logits
            ### get last non-padding index
            pred_logits = logits[index[:, 0], index[:, 1], :]
            probs = torch.softmax(pred_logits, dim=-1)  ### shape (batch, num_classes) torch.Size([4, 30522])
            ### get 0 , 1 , 2 index
            cand_probs = probs[:, cand_list]  ### torch.Size([4, 3])
            pred_labels = torch.argmax(cand_probs, dim=-1)  ### tensor([1, 1, 1, 1], device='cuda:0')
            pred_labels = cand_list[pred_labels]  ### tensor([1015, 1015, 1015, 1015], device='cuda:0')
            right_count += torch.sum(pred_labels == label).item()

        acc = right_count / len(self.eval_dataloader.dataset)
        # print(f"Accuracy: {acc}")
        wandb.log({"Validation Accuracy": acc})

    def utility_evaluate_mia(self):
        probs_lst=[]
        labels_lst=[]
        self.data_type = 'train'
        self.eval_dataloader ,train_idx,test_idx= self.get_dataloader_test_mia(self.dataset_name)
        self.model.eval()
        self.model.to(self.device)
        right_count = 0
        cand_list = self.get_candidate_list()
        cand_list = torch.tensor(cand_list).to(self.device)
        for idx, (batch_sent) in enumerate(tqdm(self.eval_dataloader)):
            inputs,tokenized_labels, index = collate(batch_sent, self.tokenizer,eval=True)
            inputs = {key: inputs[key].to(self.device) for key in inputs}
            label = torch.tensor(tokenized_labels).to(self.device)
            outputs = self.model(**inputs, return_dict=True)
            logits = outputs.logits
            ### get last non-padding index
            pred_logits = logits[index[:,0],index[:,1],:]
            probs = torch.softmax(pred_logits, dim=-1)  ### shape (batch, num_classes) torch.Size([4, 30522])
            ### get 0 , 1 , 2 index
            cand_probs = probs[:,cand_list]                     ### torch.Size([4, 3])
            pred_labels = torch.argmax(cand_probs, dim=-1)      ### tensor([1, 1, 1, 1], device='cuda:0')
            pred_labels = cand_list[pred_labels]                ### tensor([1015, 1015, 1015, 1015], device='cuda:0')
            right_count += torch.sum(pred_labels == label).item()
            probs_lst.append(cand_probs.cpu().detach())
            labels_lst.append(label.cpu().detach())

        acc = right_count / len(self.eval_dataloader.dataset)
        #print(f"Accuracy: {acc}")
        # wandb.log({"Validation Accuracy": acc})
        probs_lst=torch.cat(probs_lst)
        labels_lst=torch.cat(labels_lst)
        return {"probs":probs_lst,"labels":labels_lst,"acc":acc,"train_idx":train_idx,"test_idx":test_idx}


    ### get labels corresponding to self tokenizer
    def get_candidate_list(self):
        if self.num_labels == 2:
            candidate_list = ['0', '1']
        else:
            candidate_list = ['0', '1', '2']

        return self.tokenizer.convert_tokens_to_ids(candidate_list)


if __name__ == "__main__":
    config = config.config
    config['dataset_name'] = args.dataset_name
    config['dp'] = args.dp
    config['batch_size'] = args.batch_size
    config['model'] = args.model_name
    config['epochs'] = 1
    pprint(config)
    trainer = BERT_INFILL_tariner(config)
    ### add wanbd
    dp_str = "dp" if config["dp"] else "non_dp"
    name_ = f"{config['dataset_name']}-{config['model']}-{dp_str}"
    config_str = trainer.config_str
    name_ = name_ + str(args.seed) + '-' + config_str
    # wandb.init(
    # set the # wandb project where this run will be logged
    # project=f"llm-bert-mlm-{config['dataset_name']}",
    # name=name_,
    # # track hyperparameters and run metadata
    # config=config)
    os.makedirs(os.path.dirname(trainer.save_path), exist_ok=True)

    print(trainer.save_path + ".pt")
    # assert 0
    trainer.train_our_model()
    # res=trainer.utility_evaluate_mia()
    res = trainer.utility_evaluate_mia()
    print(trainer.save_path + str(args.seed) + ".pt")
    print("----" * 10)
    torch.save(res, trainer.save_path + str(args.seed) + ".pt")
    # run.finish()

