'''
This version of BERT is based on seqclassification.
The finetuning method is p-tuning.
'''
import wandb
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# parent_dir = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

current_dir = os.path.dirname(__file__)
parent_of_B_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))

sys.path.append(parent_of_B_dir)

import torch, transformers
from private_transformers import PrivacyEngine
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dataset.data_process import get_sent_list_cls, get_sent_list_cls_ori_order
import numpy as np
from transformers import AutoModel, AutoTokenizer
from eval.DEA.exposure_metric import calculate_single_exposure
#### example include BertForSequenceClassification ("bert-base-cased")
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
import argparse
import config
from training_interface import DP_trainer
from tqdm import tqdm
from pprint import pprint

from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    PeftConfig,
    PeftModel,
)

### overwrite privacy engine
# from opacus import PrivacyEngine

MAX_SEQ_LEN = 512


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
    # encoding = tok(sentence_a, sentence_b, padding=True, truncation=True)
    # ignore the sentence_b for -100 (None)
    if sent_b[0] == -100:
        inputs = tokenizer(sent_a, padding=True, truncation=True, return_tensors="pt")
    else:
        inputs = tokenizer(sent_a, sent_b, padding=True, truncation=True, return_tensors="pt")
    return inputs, label


class BERT_CLS_tariner_tuning(DP_trainer):
    def __init__(self, config, tuning_method="p-tuning", **kwargs):
        self.num_virtual_tokens = kwargs.get("num_virtual_tokens", 10)
        self.encoder_hidden_size = kwargs.get("encoder_hidden_size", 128)
        self.peft_config = self.get_peft_config(tuning_method)
        super().__init__(**config)
        # save checkpoints
        dp_str = "dp" if self.dp else "non_dp"
        self.tuning_method = tuning_method
        print(f'tuning method: {self.tuning_method}')
        self.config_str = f"tuning_method-{self.tuning_method}-lr-{self.lr}-batch_size-{self.batch_size}-n_accumulation_steps-{self.n_accumulation_steps}-freeze_embedding-{self.freeze_embedding}-epochs-{self.epochs}-target_epsilon-{self.target_epsilon}"
        self.save_path = os.path.join(BASE_DIR, 'checkpoints', self.config_str,
                                      self.dataset_name + '_' + self.model_name + '_' + self.tuning_method + '_' + dp_str)

    def get_peft_config(self, tuning_method):
        if tuning_method == "p-tuning":
            peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=self.num_virtual_tokens,
                                              encoder_hidden_size=self.encoder_hidden_size)
        elif tuning_method == "prefix-tuning":
            peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=self.num_virtual_tokens)
        elif tuning_method == "prompt-tuning":
            # this is for random initialization of prompt tuning (no init text given)
            peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=self.num_virtual_tokens)
        else:
            raise NotImplementedError(
                "Given tuning method is not supported, only p-tuning/prefix-tuning/prompt-tuning are supported")
        return peft_config

    def get_model(self, model):
        if (self.num_labels):
            model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=self.num_labels)
            model = get_peft_model(model, self.peft_config)
            model.print_trainable_parameters()
            self.disable_grad(model)
            model.print_trainable_parameters()

        else:
            raise NotImplementedError("num_labels is not given")
        return model

    '''
    LHR: Peft implements additional warpper ModulesToSaveWrapper:
    it wraps the classifier for peft base model (automodelforseqcls) and copies the classifier.
    Hence, both original_module and modules_to_save are set: param.requires_grad = True
    And privacy engine do not have grad sample for the original_module (Only one of them can go through the forward process insisde the wrapper)
    self.original_module = module_to_save
    self.modules_to_save = torch.nn.ModuleDict({})  --- deep copy of original_module (will be saved when call save_pretrained)
    Here I simply diasble the grad for the original_module
    '''

    def disable_grad(self, model):
        assert hasattr(model, "base_model")
        classifer = model.base_model.classifier
        modules_to_save = classifer.modules_to_save
        original_module = classifer.original_module
        ### disable grad for original_module
        for param in original_module.parameters():
            param.requires_grad = False
        # return model

    def get_tokenizer(self, model):
        print(f'load tokenizer: {model}')
        return AutoTokenizer.from_pretrained(model)

    def load_model(self, model_path):
        del self.model
        del self.peft_config
        peft_model_id = model_path
        config = PeftConfig.from_pretrained(peft_model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path,
                                                                        num_labels=self.num_labels)
        self.model = PeftModel.from_pretrained(self.model, peft_model_id)
        # self.model.eval()

    ### Get labels for corresponding dataset
    def get_dataloader(self, dataset_name):
        if dataset_name in ['qnli', "mnli", 'sst2']:
            sent_dict = get_sent_list_cls(dataset_name=dataset_name, data_type=self.data_type,SEED_NUMBER=args.seed)
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

    def get_dataloader_test_mia(self, dataset_name):
        if dataset_name in ['qnli', "mnli", 'sst2']:
            sent_dict,train_index,test_index = get_sent_list_cls_ori_order(dataset_name=dataset_name, data_type="train",SEED_NUMBER=args.seed)
            data = CLS_dataset(sent_dict)
            self.num_labels = sent_dict["label_num"]
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
        self.model.to(self.device)
        print(f"LM: {self.model_name} is loaded")
        for epoch in range(self.epochs):
            ##### training code #####
            self.model.train()
            for idx, (sent_a, sent_b, label) in enumerate(tqdm(self.dataloader)):
                inputs, label = collate(sent_a, sent_b, label, self.tokenizer)
                ### skip long sequence after adding virtual tokens (seq+len(virtual) > 512)
                if (inputs['input_ids'].shape[1] + self.num_virtual_tokens + 1 > MAX_SEQ_LEN):
                    continue
                inputs = {key: inputs[key].to(self.device) for key in inputs}
                label = torch.tensor(label).to(self.device)
                outputs = self.model(**inputs, labels=label, return_dict=True)
                logits = outputs.logits
                loss = F.cross_entropy(logits, label, reduction="none")
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
                    # self.optimizer.step()
                wandb.init()
                wandb.log({"batch loss": loss.mean(), "built-in loss": outputs.loss})
                # print(f"{epoch + 1}th epoch{idx}th batch output loss: {outputs.loss}. CE loss: {torch.mean(loss)}")
            # save checkpoints
            save_path = self.save_path
            print(f'save model to {save_path}')
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
            if (inputs['input_ids'].shape[1] + self.num_virtual_tokens + 1 > MAX_SEQ_LEN):
                continue
            inputs = {key: inputs[key].to(self.device) for key in inputs}
            labels = torch.tensor(labels).to(self.device)
            outputs = self.model(**inputs, labels=labels, return_dict=True)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            right_count += torch.sum(preds == labels).item()
        acc = right_count / len(self.eval_dataloader.dataset)
        print(f"Accuracy: {acc}")
        wandb.log({"Validation Accuracy": acc})

    @torch.no_grad()
    def utility_evaluate_mia(self):
        # self.data_type = 'dev'
        probs_lst=[]
        labels_lst=[]
        self.eval_dataloader,train_idx,test_idx = self.get_dataloader_test_mia(self.dataset_name)
        self.model.eval()
        self.model.to(self.device)
        right_count = 0
        for idx, (sent_a, sent_b, labels) in enumerate(tqdm(self.eval_dataloader)):
            inputs, labels = collate(sent_a, sent_b, labels, self.tokenizer)
            # if(inputs['input_ids'].shape[1] + self.num_virtual_tokens + 1> MAX_SEQ_LEN):
            #         continue
            inputs = {key: inputs[key].to(self.device) for key in inputs}
            labels = torch.tensor(labels).to(self.device)
            outputs = self.model(**inputs, labels=labels,return_dict=True)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            right_count += torch.sum(preds == labels).item()
            probs_lst.append(probs.cpu().detach().numpy())
            labels_lst.append(labels.cpu().detach().numpy())
        acc = right_count / len(self.eval_dataloader.dataset)
        print(f"Accuracy: {acc}")
        # wandb.log({"Validation Accuracy": acc})
        return {"probs":probs_lst,"labels":labels_lst,"acc":acc,"train_idx":train_idx,"test_idx":test_idx}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training external NN as baselines')
    parser.add_argument('--tuning_method', type=str, default="prefix-tuning",
                        help='tuning_method:only p-tuning/prefix-tuning/prompt-tuning are supported')
    parser.add_argument('--num_virtual_tokens', type=int, default=15, help='num_virtual_tokens')
    parser.add_argument('--encoder_hidden_size', type=int, default=128, help='encoder_hidden_size')
    parser.add_argument('--seed', type=int, default=42, help='dataset name')
    parser.add_argument('--model_name', type=str, default="", help='dataset name')
    parser.add_argument('--batch_size', type=int, default=16, help='dataset name')
    parser.add_argument('--dataset_name', type=str, default='qnli', help='model name')
    parser.add_argument('--dp', type=bool, default=False, help='whether to use dp')
    args = parser.parse_args()
    print(args)
    # assert 0
    # tuning config
    tuning_config = {}
    tuning_config["tuning_method"] = args.tuning_method
    tuning_config["num_virtual_tokens"] = args.num_virtual_tokens
    tuning_config["encoder_hidden_size"] = args.encoder_hidden_size

    # trainer config overwrite
    config = config.config
    config['lr'] = 1e-2
    config['epochs'] = 10

    config["seed"] = args.seed
    config["model"] = args.model_name
    config["dataset_name"] = args.dataset_name
    config["dp"] = args.dp
    config["batch_size"] = args.batch_size
    config['n_accumulation_steps'] = int(config['virtual_batch_size'] / config['batch_size'])


    print("Config is loaded")
    pprint(config)
    # assert 0
    trainer = BERT_CLS_tariner_tuning(config, **tuning_config)
    ### add wanbd
    dp_str = "dp" if config["dp"] else "non_dp"
    name_ = f"{config['dataset_name']}-{config['model']}-{dp_str}"
    config_str = trainer.config_str
    name_ = name_ + '-' + config_str

    os.makedirs(os.path.dirname(trainer.save_path), exist_ok=True)
    trainer.train_our_model()
    res = trainer.utility_evaluate_mia()
    print(trainer.save_path + str(args.seed) + ".pt")
    print("----" * 10)
    torch.save(res, trainer.save_path + str(args.seed) + ".pt")


"""
for i in {0..6}
do
for g in {0..9}
do
CUDA_VISIBLE_DEVICES=$g python examples/bert_cls_p-tuning.py --seed ${i}${g} --dp False  --dataset_name qnli --batch_size 8  &
done
wait
done

"""
