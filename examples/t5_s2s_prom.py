import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import config
from training_interface import DP_trainer
from peft import PromptTuningConfig, TaskType, get_peft_model, PeftModel, PeftConfig
import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader, Dataset
from dataset.data_process import get_sent_list
import torch.nn.functional as F
import pickle
import random
from random import sample

from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType

MAX_SEQ_LEN = 1024

with open(os.path.join("..", "dataset", "default_canary.json"), "rb") as f:
    default_canary = pickle.load(f)


def random_insert_seq(lst, seq):
    insert_locations = sample(range(len(lst) + len(seq)), len(seq))
    inserts = dict(zip(insert_locations, seq))
    input = iter(lst)
    lst[:] = [inserts[pos] if pos in inserts else next(input)
              for pos in range(len(lst) + len(seq))]
    return lst, insert_locations


def generate_evaluation_sequence_for_t5(canary_type="email", use_full_text=False, insert_proportion=0.4):
    canary_format = default_canary[canary_type]["canary_format"]
    fill_list = default_canary[canary_type]["fill_list"]
    length = round(len(fill_list) * insert_proportion)
    inserted = fill_list[:length]
    not_inserted = fill_list[length:]
    if use_full_text:
        inserted = [canary_format.replace('*', fill).split(" <SEP> ") for fill in inserted]
        not_inserted = [canary_format.replace('*', fill).split(" <SEP> ") for fill in not_inserted]
    return inserted, not_inserted


canary_type_list = ['name', 'city', 'email', 'phone_number', 'letters', 'setting_1', 'setting_2']
insert_proportion_list = [0.4] * 7
insert_time_base_list = [10] * 7

t5_canary_format_list = ["What is your name? <SEP> My name is *",
                         "Where do you live? <SEP> I live in *. It is a beautiful city",
                         "What is your email address? <SEP> My email is *",
                         "What is your phone number? <SEP> My phone number is*. You can call me by this number.",
                         "What is your password? <SEP> The letters are*. It is my password.",
                         "Choose one random word. <SEP> The chosen word is *",
                         "Choose three random words. <SEP> The chosen words are *"
                         ]

for canary_type, new_format in zip(canary_type_list, t5_canary_format_list):
    default_canary[canary_type]['canary_format'] = new_format

class CanaryDatasetForT5(Dataset):
    def __init__(self, canary_type_list, insert_proportion_list, insert_time_base_list, dataset_name,
                 data_type="train"):
        self.canary_type_list = canary_type_list
        self.insert_proportion_list = insert_proportion_list
        self.insert_time_base_list = insert_time_base_list
        self.dataset_name = dataset_name
        if self.dataset_name in ['qnli', "mnli", 'sst2']:
            data_list = get_sent_list(dataset_name=dataset_name, data_type=data_type)
        else:
            raise NotImplementedError("Given dataset is not supported now")
        t5_data_list = []
        for data in data_list:
            data_split = data.split(' <SEP> ')
            t5_data = []
            t5_data.append(' <SEP> '.join(data_split[:-1]))
            t5_data.append(data_split[-1][-1] + ' </s>')
            t5_data_list.append(t5_data)
        self.data_list = t5_data_list
        self.canary_format_list = []
        self.insert_time_list = []
        self.canary_infill_list = []
        self.canary_text_list = []
        for canary_type, insert_proportion, insert_time_base in zip(canary_type_list,
                                                                    insert_proportion_list, insert_time_base_list):
            canary_format = default_canary[canary_type]["canary_format"]
            fill_list = default_canary[canary_type]["fill_list"]
            length = round(len(fill_list) * insert_proportion)
            fill_list = fill_list[:length]
            insert_time_list = [i * insert_time_base for i in range(1, length + 1)]
            canary_text_list = [canary_format.replace('*', fill).split(' <SEP> ') for fill in fill_list]
            self.insert_time_list.extend(insert_time_list)
            self.canary_format_list.extend([canary_format] * length)
            self.canary_infill_list.extend(fill_list)
            self.canary_text_list.extend(canary_text_list)

        self.canary_loc_list = []
        self.insert_canary()


    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)

    def insert_canary(self):
        insert_canary_text_list = []
        if self.dataset_name == "personachat":
            for canary_text, insert_time in zip(self.canary_text_list, self.insert_time_list):
                insert_canary_text_list.extend([canary_text] * insert_time)
            idx = random.sample(range(len(self.data_list)), len(insert_canary_text_list))
            for i_0, canary_text in zip(idx, insert_canary_text_list):
                sentence = self.data_list[i_0]
                dialog = sentence.split(" <SEP> ")
                i_1 = random.randint(0, len(dialog))
                self.canary_loc_list.append((i_0, i_1))
                dialog.insert(i_1, canary_text)
                self.data_list[i_0] = " <SEP> ".join(dialog)
        else:
            for canary_text, insert_time in zip(self.canary_text_list, self.insert_time_list):
                insert_canary_text_list.extend([canary_text] * insert_time)
            self.data_list, self.canary_loc_list = random_insert_seq(self.data_list, insert_canary_text_list)

class T5_Dataset(Dataset):
    def __init__(self, dataset_name, data_type='train'):
        if dataset_name in ['mnli', 'qnli', 'sst2']:
            data_list = get_sent_list(dataset_name=dataset_name, data_type=data_type)
        else:
            raise NotImplementedError
        t5_data_list = []
        for data in data_list:
            data_split = data.split(' <SEP> ')
            t5_data = []
            t5_data.append(' <SEP> '.join(data_split[:-1]))
            t5_data.append(data_split[-1][-1] + ' </s>')
            t5_data.append(data_split[-1][-1])
            t5_data_list.append(t5_data)
        self.data_list = t5_data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


class T5_S2S_Prompt(DP_trainer):
    def __init__(self, config, **kwargs):

        self.num_virtual_token = kwargs.get("num_virtual_token", 15)
        self.adapter_name = kwargs.get("adapter_name", "t5_s2s_prom")
        self.dataset_name = config["dataset_name"]
        if self.dataset_name == "qnli":
            self.prompt_tuning_init_text = "Classify if the relation of these two sentence is entailment or not"
        elif self.dataset_name == "sst2":
            self.prompt_tuning_init_text = "Classify if the sentiment is negative or positive"
        elif self.dataset_name == "mnli":
            self.prompt_tuning_init_text = "Classify if the relation of these two sentence is entailment, neutral or contradiction"
        else:
            self.prompt_tuning_init_text = "The text is a dialog between peoples"

        super().__init__(**config)

        dp_str = "dp" if self.dp else "non_dp"
        self.num_decode_virtual_tokens = 0
        self.tuning_method = "prompt-tuning"

        self.config_str = f"tuning_method-{self.tuning_method}-lr-{self.lr}-batch_size-{self.batch_size}-n_accumulation_steps-{self.n_accumulation_steps}-freeze_embedding-{self.freeze_embedding}-epochs-{self.epochs}-target_epsilon-{self.target_epsilon}"
        if len(self.canary_type_list) > 0:
            self.config_str = "canary_inserted-" + self.config_str
        self.save_path = os.path.join(BASE_DIR, 'checkpoints', self.config_str,
                                      self.dataset_name + '_' + self.model_name + '_' + self.tuning_method + '_' + dp_str)

    def get_tokenizer(self, model):
        # tokenizer = T5Tokenizer.from_pretrained(model)
        tokenizer = T5Tokenizer.from_pretrained(model)
        return tokenizer

    def get_model(self, model):
        # model = T5ForConditionalGeneration(model)
        model = T5ForConditionalGeneration.from_pretrained(model)
        peft_config = PromptTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM,
                                         inference_mode=False,
                                         num_virtual_tokens=self.num_virtual_token,
                                         prompt_tuning_init_text=self.prompt_tuning_init_text,
                                         tokenizer_name_or_path=self.model_name
                                         )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model

    def get_dataloader(self, dataset_name):
        if dataset_name in ['mnli', 'qnli', 'sst2']:
            if len(self.canary_type_list) == 0:
                data = T5_Dataset(dataset_name=dataset_name, data_type=self.data_type)
            else:
                data = CanaryDatasetForT5(canary_type_list=self.canary_type_list,
                                          insert_proportion_list=self.insert_proportion_list,
                                          insert_time_base_list=self.insert_time_base_list,
                                          dataset_name=dataset_name,
                                          )
        else:
            raise NotImplementedError

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

    def train_on_batch(self, batch_text, model, tokenizer, criterion, device):
        batch_source_text = list(batch_text[0])
        batch_target_text = list(batch_text[1])

        # tokenize
        tokenizer.pad_token = tokenizer.eos_token
        if self.tokenizer_len:
            tokenized_source = tokenizer(batch_source_text, return_tensors='pt', padding='max_length', truncation=True,
                                         max_length=self.tokenizer_len)
        else:
            tokenized_source = tokenizer(batch_source_text, return_tensors='pt', padding=True, truncation=True)

        if len(self.canary_type_list) == 0:
            tokenized_target = tokenizer(batch_target_text, return_tensors='pt', truncation=True, max_length=2)
        else:
            tokenized_target = tokenizer(batch_target_text, return_tensors='pt', padding='longest', truncation=True, max_length=25)

        input_ids = tokenized_source['input_ids'].to(device)
        src_attention_mask = tokenized_source['attention_mask'].to(device)

        labels = tokenized_target['input_ids'].to(device)

        tgt_attention_mask = tokenized_target['attention_mask'].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=src_attention_mask,
            labels=labels
        ).logits


        if self.dp:
            loss = criterion(logits, labels, mask=tgt_attention_mask, label_smoothing=0.02, reduce=True)
        else:
            loss = criterion(logits, labels, mask=tgt_attention_mask, label_smoothing=0.02, reduce="batch")
        return loss

    def utility_evaluate(self):
        self.data_type = 'dev'
        self.eval_dataloader = self.get_dataloader(self.dataset_name)
        self.model.eval()
        self.model.to(self.device)
        length = len(self.eval_dataloader.dataset)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        right_count = 0
        for batch_text in self.eval_dataloader:
            batch_right_count = self.s2s_utility_prediction_for_batch(self.model, self.tokenizer, batch_text,
                                                                      self.dataset_name, self.device,
                                                                      num_virtual_token=self.num_decode_virtual_tokens
                                                                      )
            right_count += batch_right_count
        acc = right_count/length
        print(acc)
        # wandb.log({"Validation Accuracy": acc})

    def s2s_utility_prediction_for_batch(self, model, tokenizer, batch_text, dataset_name, device='cuda',
                                         num_virtual_token=0):
        batch_right_count = 0
        if dataset_name == 'qnli':
            prediction_range = ['0 </s>', '1 </s>']
        elif dataset_name == 'sst2':
            prediction_range = ['0 </s>', '1 </s>']
        elif dataset_name == 'mnli':
            prediction_range = ['0 </s>', '1 </s>', '2 </s>']
        else:
            raise NotImplementedError
        prediction_idx_range = []
        for i in prediction_range:
            prediction_idx_range.append(tokenizer.encode(i)[0])

        # prepare for tokenize
        batch_source_text = list(batch_text[0])
        batch_target_text = list(batch_text[1])
        #
        # tokenize
        with torch.no_grad():
            tokenizer.pad_token = tokenizer.eos_token
            if self.tokenizer_len:
                tokenized_source = tokenizer(batch_source_text, return_tensors='pt', padding='max_length',
                                             truncation=True,
                                             max_length=self.tokenizer_len)
            else:
                tokenized_source = tokenizer(batch_source_text, return_tensors='pt', padding=True, truncation=True)

            tokenized_target = tokenizer(batch_target_text, return_tensors='pt', truncation=True, max_length=2)

            input_ids = tokenized_source['input_ids'].to(device)
            src_attention_mask = tokenized_source['attention_mask'].to(device)

            labels = tokenized_target['input_ids'].to(device)

            full_logits = model(
                input_ids=input_ids,
                attention_mask=src_attention_mask,
                labels=labels
            ).logits

            logits = full_logits[:, num_virtual_token, prediction_idx_range]

            batch_label = labels[:, 0].tolist()
            prob = F.softmax(logits)
            pred_idx_list = torch.argmax(prob, dim=-1).tolist()
            for pred_idx, label in zip(pred_idx_list, batch_label):
                if prediction_idx_range[pred_idx] == label:
                    batch_right_count += 1
        return batch_right_count

    def save_checkpoints(self):
        self.model.save_pretrained(self.save_path)

    def load_model(self, model_path):
        del self.model
        base_model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.model = PeftModel.from_pretrained(model=base_model, model_id=model_path)

if __name__ == "__main__":
    config = config.config
    config['model'] = "google-t5/t5-small"
    config['lr'] = 1e-2
    config['dataset_name'] = 'sst2'
    config['dp'] = True

    tuning_config = {}
    tuning_config["num_virtual_token"] = 15
    tuning_config["adapter_name"] = "t5_s2s_prom"

    trainer = T5_S2S_Prompt(config, **tuning_config)

    trainer.train_our_model()

